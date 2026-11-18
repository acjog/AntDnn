// ---------------------------------------------------------------------------
// SLANet_plus 完整网络的 AntDnn 演示 (encoder + decoder)
//
// 用 AntDnn 的 Tensor + 算子跑通 SLANet_plus 的完整前向:
//   Backbone: PPLCNet   (det_pp_lcnet.py)   -> 4 尺度特征
//   Neck:     CSPPAN    (csp_pan.py)        -> 4 尺度特征 (通道 96)
//   Head:     SLAHead   (table_att_head.py) -> 自回归 Attention-GRU 解码
//
// 内存布局: 全程 NHWC (输入 [H, W, C], 无 batch 维)
// 权重: encoder.antlist (由 export_encoder.py 从 encoder.onnx 转换, NCHW->NHWC)
//       weights.antlist (decoder, 由 gen_weights.py 生成)
//
// 网络结构:
//   PPLCNet:
//     conv1 (3x3, 3->16, s2) -> blocks2~6 (depthwise separable)
//     输出 4 尺度: [122,122,64] [61,61,128] [31,31,256] [16,16,512]
//   CSPPAN: Channel_T -> top-down(3 CSPLayer) -> bottom-up(3 CSPLayer)
//     输出 4 尺度: 通道 96, 最后一个 [16,16,96] 展平成 [256, 96]
//   SLAHead (decoder, 自回归):
//     AttentionGRUCell(i2h/h2h/score + GRU) + structure/loc generator
//     每步 argmax 选 token, 遇 EOS 停止
// ---------------------------------------------------------------------------

#include <antdnn/Tensor.h>
#include <antdnn/Activation.h>
#include <antdnn/Indexing.h>
#include "pplcnet.h"
#include "csppan.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

using namespace antdnn;
using namespace std;

// ---- 维度常量 (与真实 SLANet_plus 一致) ----
static const int ENC_SEQ = 256;   // 16x16 (CSPPAN 最后尺度展平)
static const int ENC_FEAT = 96;   // CSPPAN 输出通道
static const int DEC_HID = 256;
static const int NUM_CLS = 50;    // 结构字典大小 (table_structure_dict)
static const int LOC_NUM = 8;     // loc_reg_num
static const int GRU_IN = ENC_FEAT + NUM_CLS; // 146
static const int EOS = NUM_CLS - 1;

// ---- structure 字典 (与 table_structure_dict_ch.txt + merge_no_span_structure 一致) ----
// 顺序: [0]=sos, [1..48]=结构标签, [49]=eos
static const char* STRUCTURE_DICT[NUM_CLS] = {
    "sos",            // 0
    "<thead>",        // 1
    "</thead>",       // 2
    "<tbody>",        // 3
    "</tbody>",       // 4
    "<tr>",           // 5
    "</tr>",          // 6
    "<td",            // 7
    ">",              // 8
    "</td>",          // 9
    " colspan=\"2\"",   // 10
    " colspan=\"3\"",   // 11
    " colspan=\"4\"",   // 12
    " colspan=\"5\"",   // 13
    " colspan=\"6\"",   // 14
    " colspan=\"7\"",   // 15
    " colspan=\"8\"",   // 16
    " colspan=\"9\"",   // 17
    " colspan=\"10\"",  // 18
    " colspan=\"11\"",  // 19
    " colspan=\"12\"",  // 20
    " colspan=\"13\"",  // 21
    " colspan=\"14\"",  // 22
    " colspan=\"15\"",  // 23
    " colspan=\"16\"",  // 24
    " colspan=\"17\"",  // 25
    " colspan=\"18\"",  // 26
    " colspan=\"19\"",  // 27
    " colspan=\"20\"",  // 28
    " rowspan=\"2\"",   // 29
    " rowspan=\"3\"",   // 30
    " rowspan=\"4\"",   // 31
    " rowspan=\"5\"",   // 32
    " rowspan=\"6\"",   // 33
    " rowspan=\"7\"",   // 34
    " rowspan=\"8\"",   // 35
    " rowspan=\"9\"",   // 36
    " rowspan=\"10\"",  // 37
    " rowspan=\"11\"",  // 38
    " rowspan=\"12\"",  // 39
    " rowspan=\"13\"",  // 40
    " rowspan=\"14\"",  // 41
    " rowspan=\"15\"",  // 42
    " rowspan=\"16\"",  // 43
    " rowspan=\"17\"",  // 44
    " rowspan=\"18\"",  // 45
    " rowspan=\"19\"",  // 46
    " rowspan=\"20\"",  // 47
    "<td></td>",       // 48
    "eos",             // 49
};

// ---- 简单矩阵/向量运算辅助 (直接操作裸指针, 符合 AntDnn 风格) ----

// y = x @ W  (x: [m], W: [m, n], 行主序) -> y: [n]
static void matvec(const float *x, const float *W, float *y, int m, int n)
{
    for (int j = 0; j < n; j++)
    {
        float s = 0.f;
        for (int i = 0; i < m; i++)
            s += x[i] * W[i * n + j];
        y[j] = s;
    }
}

// Y = X @ W  (X: [r, m], W: [m, n]) -> Y: [r, n]
static void matmul(const float *X, const float *W, float *Y, int r, int m, int n)
{
    for (int i = 0; i < r; i++)
        matvec(X + i * m, W, Y + i * n, m, n);
}

static float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
static float my_tanh(float x) { return tanhf(x); }

// ---- 权重容器 ----
struct DecoderWeights
{
    vector<float> i2h_w, h2h_w, h2h_b, score_w;
    vector<float> gru_wx, gru_wh, gru_bx, gru_bh;
    vector<float> fc1_w, fc1_b, struct_w, struct_b;
    vector<float> fc2_w, fc2_b, loc_w, loc_b;
};

// 从 Tensor 拷贝到 vector<float>
static void to_vec(const Tensor &ts, vector<float> &v)
{
    v.resize(ts.size());
    memcpy(v.data(), ts.ptr_read(), ts.size() * sizeof(float));
}

// 加载 16 个权重张量到结构体 (顺序与 gen_weights.py 一致)
static bool load_decoder_weights(const vector<Tensor> &tsvec, DecoderWeights &w)
{
    if (tsvec.size() != 16)
        return false;
    to_vec(tsvec[0], w.i2h_w);
    to_vec(tsvec[1], w.h2h_w);
    to_vec(tsvec[2], w.h2h_b);
    to_vec(tsvec[3], w.score_w);
    to_vec(tsvec[4], w.gru_wx);
    to_vec(tsvec[5], w.gru_wh);
    to_vec(tsvec[6], w.gru_bx);
    to_vec(tsvec[7], w.gru_bh);
    to_vec(tsvec[8], w.fc1_w);
    to_vec(tsvec[9], w.fc1_b);
    to_vec(tsvec[10], w.struct_w);
    to_vec(tsvec[11], w.struct_b);
    to_vec(tsvec[12], w.fc2_w);
    to_vec(tsvec[13], w.fc2_b);
    to_vec(tsvec[14], w.loc_w);
    to_vec(tsvec[15], w.loc_b);
    return true;
}

// ---- AttentionGRUCell 单步 ----
// prev_hidden: [DEC_HID], batch_H: [ENC_SEQ, ENC_FEAT], char_onehot: [NUM_CLS]
// 返回新的 hidden [DEC_HID]
static void attention_gru_cell(
    const float *prev_hidden, const float *batch_H, const float *char_onehot,
    const DecoderWeights &w, float *h_next)
{
    // i2h: batch_H @ i2h_w -> [ENC_SEQ, DEC_HID]
    vector<float> batch_H_proj(ENC_SEQ * DEC_HID);
    matmul(batch_H, w.i2h_w.data(), batch_H_proj.data(), ENC_SEQ, ENC_FEAT, DEC_HID);

    // h2h: prev_hidden @ h2h_w + h2h_b -> [DEC_HID]
    vector<float> prev_hidden_proj(DEC_HID);
    matvec(prev_hidden, w.h2h_w.data(), prev_hidden_proj.data(), DEC_HID, DEC_HID);
    for (int i = 0; i < DEC_HID; i++)
        prev_hidden_proj[i] += w.h2h_b[i];

    // res = tanh(batch_H_proj + prev_hidden_proj)
    vector<float> res(ENC_SEQ * DEC_HID);
    for (int i = 0; i < ENC_SEQ; i++)
        for (int j = 0; j < DEC_HID; j++)
            res[i * DEC_HID + j] = my_tanh(batch_H_proj[i * DEC_HID + j] + prev_hidden_proj[j]);

    // e = res @ score_w -> [ENC_SEQ, 1]
    vector<float> e(ENC_SEQ);
    matmul(res.data(), w.score_w.data(), e.data(), ENC_SEQ, DEC_HID, 1);

    // softmax(e, axis=0)
    float maxv = e[0];
    for (int i = 1; i < ENC_SEQ; i++) if (e[i] > maxv) maxv = e[i];
    vector<float> alpha(ENC_SEQ);
    float sum = 0.f;
    for (int i = 0; i < ENC_SEQ; i++) { alpha[i] = expf(e[i] - maxv); sum += alpha[i]; }
    for (int i = 0; i < ENC_SEQ; i++) alpha[i] /= sum;

    // context = alpha^T @ batch_H -> [ENC_FEAT]
    vector<float> context(ENC_FEAT, 0.f);
    for (int i = 0; i < ENC_SEQ; i++)
        for (int j = 0; j < ENC_FEAT; j++)
            context[j] += alpha[i] * batch_H[i * ENC_FEAT + j];

    // concat(context, char_onehot) -> [GRU_IN]
    vector<float> concat(GRU_IN);
    memcpy(concat.data(), context.data(), ENC_FEAT * sizeof(float));
    memcpy(concat.data() + ENC_FEAT, char_onehot, NUM_CLS * sizeof(float));

    // GRU: gates_x = concat @ gru_wx + gru_bx, gates_h = prev_hidden @ gru_wh + gru_bh
    vector<float> gates_x(3 * DEC_HID), gates_h(3 * DEC_HID);
    matvec(concat.data(), w.gru_wx.data(), gates_x.data(), GRU_IN, 3 * DEC_HID);
    matvec(prev_hidden, w.gru_wh.data(), gates_h.data(), DEC_HID, 3 * DEC_HID);
    for (int i = 0; i < 3 * DEC_HID; i++) { gates_x[i] += w.gru_bx[i]; gates_h[i] += w.gru_bh[i]; }

    for (int i = 0; i < DEC_HID; i++)
    {
        float r = sigmoid(gates_x[i] + gates_h[i]);
        float z = sigmoid(gates_x[DEC_HID + i] + gates_h[DEC_HID + i]);
        float n = my_tanh(gates_x[2 * DEC_HID + i] + r * gates_h[2 * DEC_HID + i]);
        h_next[i] = (prev_hidden[i] - n) * z + n;
    }
}

// ---- 图片预处理 (与 PaddleOCR SLANet_plus 的 transform 一致) ----
// 流程: 读图(BGR) -> 等比缩放(max_len=488) -> Normalize((x/255-mean)/std) -> Pad(488x488)
// 输入: 图片路径
// 输出: input Tensor [488, 488, 3] (NHWC, 已归一化)
//       ratio (缩放比例), src_img (原图 BGR, 用于画框可视化)
// 返回: false 表示读图失败
static bool preprocess_image(const std::string& img_path, Tensor& input,
                             float& ratio, cv::Mat& src_img)
{
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);  // BGR
    if (img.empty())
    {
        cerr << "无法读取图片: " << img_path << endl;
        return false;
    }

    int h = img.rows, w = img.cols;

    // 1. ResizeTableImage: 按长边等比缩放到 488
    const int MAX_LEN = 488;
    ratio = (float)MAX_LEN / (float)std::max(h, w);
    int resize_h = (int)(h * ratio);
    int resize_w = (int)(w * ratio);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    cout << "原始图片: " << w << "x" << h
         << " -> resize: " << resize_w << "x" << resize_h
         << " (ratio=" << ratio << ")" << endl;

    // 保存原图 (用于画框可视化)
    src_img = img;

    // 2. NormalizeImage (order='hwc', 即 NHWC, 保持 BGR):
    //    (x/255 - mean) / std, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    //    注意: PaddleOCR 用 img_mode=BGR, mean/std 直接按 BGR 通道顺序套用
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float stdv[3] = { 0.229f, 0.224f, 0.225f };

    // 3. PaddingTableImage: pad 到 488x488 (右下方补 0)
    int in_shape[3] = { 488, 488, 3 };
    input.recreate(3, in_shape);
    input.set_to(0.0f);
    float* dst = input.ptr_write();

    for (int y = 0; y < resize_h; y++)
    {
        const float* row = resized.ptr<float>(y);
        float* dst_row = dst + y * 488 * 3;
        for (int x = 0; x < resize_w; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = row[x * 3 + c];
                dst_row[x * 3 + c] = (v - mean[c]) / stdv[c];
            }
        }
        // 右侧 padding 区域保持 0 (已 set_to(0))
    }
    // 下方 padding 区域保持 0

    cout << "预处理完成: [488, 488, 3]" << endl;
    return true;
}

int main(int argc, char** argv)
{
    // ---- 1. 加载 decoder 权重 ----
    vector<Tensor> tsvec;
    load_tensor_vector("weights.antlist", tsvec);
    DecoderWeights w;
    if (!load_decoder_weights(tsvec, w))
    {
        cerr << "权重加载失败 (expect 16, got " << tsvec.size() << ")." << endl;
        cerr << "请先运行: python3 gen_weights.py" << endl;
        return 1;
    }
    cout << "加载 decoder 权重: " << tsvec.size() << " 个张量" << endl;

    // ---- 2. encoder 前向 (PPLCNet + CSPPAN) ----
    // 2a. 加载 encoder 权重 (encoder.antlist, 由 export_pdparams.py 生成)
    vector<Tensor> enc_weights;
    load_tensor_vector("encoder.antlist", enc_weights);

    // 2b. 准备输入 (488x488x3, NHWC)
    Tensor input;
    float ratio = 1.0f;
    cv::Mat src_img;   // 原图 (用于画框可视化)
    bool has_image = false;
    if (argc > 1)
    {
        // 从命令行读真实图片并预处理
        if (!preprocess_image(argv[1], input, ratio, src_img))
            return 1;
        has_image = true;
    }
    else
    {
        // 未指定图片时, 用随机数据演示
        cout << "未指定图片, 使用随机输入演示 (用法: ./slanet_plus_demo <图片路径>)" << endl;
        int in_shape[3] = { 488, 488, 3 };
        input.recreate(3, in_shape);
        input.set_to(0.0f);
        float* p = input.ptr_write();
        srand(12345);
        for (int i = 0; i < input.size(); i++)
            p[i] = (rand() % 1000) / 1000.0f;
    }

    // 2c. 跑 encoder, 得到特征 batch_H [256, 96]
    Tensor batch_H_ts;
    const float* batch_H = nullptr;

    if (enc_weights.size() > 0)
    {
        // 用真实 encoder 权重跑 PPLCNet + CSPPAN
        cout << "encoder 权重: " << enc_weights.size() << " 个张量" << endl;
        slanet::PPLCNet backbone(&enc_weights);
        std::vector<Tensor> feats;
        backbone.forward(input, feats);
        cout << "PPLCNet 输出 " << feats.size() << " 个尺度:" << endl;
        for (size_t i = 0; i < feats.size(); i++)
            cout << "  [" << feats[i].shape()[0] << ", " << feats[i].shape()[1]
                 << ", " << feats[i].shape()[2] << "]" << endl;

        slanet::CSPPAN neck(&enc_weights, backbone.cursor);
        std::vector<Tensor> neck_outs;
        neck.forward(feats, neck_outs);
        cout << "CSPPAN 输出 " << neck_outs.size() << " 个尺度:" << endl;
        for (size_t i = 0; i < neck_outs.size(); i++)
            cout << "  [" << neck_outs[i].shape()[0] << ", " << neck_outs[i].shape()[1]
                 << ", " << neck_outs[i].shape()[2] << "]" << endl;

        // 取最后一个尺度 [16,16,96] 展平成 [256, 96]
        Tensor& last = neck_outs.back();
        batch_H_ts = last;  // 数据是连续的 [H,W,C], 展平后就是 [H*W, C]
        batch_H = batch_H_ts.ptr_read();
        cout << "encoder 特征: [" << last.shape()[0]*last.shape()[1] << ", " << last.shape()[2] << "]" << endl;
    }
    else
    {
        // 降级: 没有 encoder.antlist 时用模拟特征 (需 gen_weights.py 生成 enc_feat.antts)
        Tensor enc_ts;
        if (!enc_ts.load_file("enc_feat.antts"))
        {
            cerr << "encoder 权重和模拟特征都不存在。" << endl;
            cerr << "请运行: python3 export_encoder.py (生成 encoder.antlist)" << endl;
            cerr << "    或: python3 gen_weights.py (生成 enc_feat.antts 模拟特征)" << endl;
            return 1;
        }
        batch_H_ts = enc_ts;
        batch_H = batch_H_ts.ptr_read();
        cout << "使用模拟 encoder 特征 shape: [" << enc_ts.shape()[0] << ", " << enc_ts.shape()[1] << "]" << endl;
    }

    // ---- 3. 自回归解码 ----
    // 用 AntDnn 的 Tensor 保存 hidden 与 one-hot (演示算子用法)
    vector<float> hidden(DEC_HID, 0.f);
    int pre_char = 0;

    vector<int> tokens;
    vector<vector<float>> structs, locs;

    const int MAX_STEPS = 500; // 与 SLANet_plus.yml 的 max_text_length 一致
    for (int step = 0; step < MAX_STEPS; step++)
    {
        // one-hot 编码当前 token (用 AntDnn 的 one_hot 算子)
        int shape1[1] = { 1 };
        Tensor pre_char_ts(1, shape1);
        pre_char_ts.set_to((float)pre_char);

        Tensor onehot_ts;
        one_hot(pre_char_ts, onehot_ts, NUM_CLS); // [1, NUM_CLS]
        const float *char_onehot = onehot_ts.ptr_read();

        // AttentionGRUCell 单步
        vector<float> h_next(DEC_HID);
        attention_gru_cell(hidden.data(), batch_H, char_onehot, w, h_next.data());
        hidden = h_next;

        // structure_generator: hidden -> fc1 -> struct (softmax)
        vector<float> h(DEC_HID);
        matvec(hidden.data(), w.fc1_w.data(), h.data(), DEC_HID, DEC_HID);
        for (int i = 0; i < DEC_HID; i++) h[i] += w.fc1_b[i];

        vector<float> struct_logit(NUM_CLS);
        matvec(h.data(), w.struct_w.data(), struct_logit.data(), DEC_HID, NUM_CLS);
        for (int i = 0; i < NUM_CLS; i++) struct_logit[i] += w.struct_b[i];

        // softmax + argmax (用 AntDnn 的 activation_softmax 与 argmax 算子)
        int sshape[1] = { NUM_CLS };
        Tensor struct_logit_ts(1, sshape, struct_logit.data());
        Tensor struct_prob_ts;
        struct_prob_ts = struct_logit_ts;
        activation_softmax(struct_prob_ts);   // 原地 softmax
        const float *struct_prob = struct_prob_ts.ptr_read();

        Tensor argmax_ts;
        argmax(struct_logit_ts, argmax_ts);   // argmax 选 token
        pre_char = (int)argmax_ts.ptr_read()[0];

        // loc_generator: hidden -> fc2 -> loc (sigmoid)
        vector<float> h2(DEC_HID);
        matvec(hidden.data(), w.fc2_w.data(), h2.data(), DEC_HID, DEC_HID);
        for (int i = 0; i < DEC_HID; i++) h2[i] += w.fc2_b[i];

        vector<float> loc(LOC_NUM);
        matvec(h2.data(), w.loc_w.data(), loc.data(), DEC_HID, LOC_NUM);
        for (int i = 0; i < LOC_NUM; i++) loc[i] += w.loc_b[i];

        // sigmoid (用 AntDnn 的 hardsigmoid? 不, 这里用标准 sigmoid 手写)
        for (int i = 0; i < LOC_NUM; i++) loc[i] = sigmoid(loc[i]);

        tokens.push_back(pre_char);
        structs.push_back(vector<float>(struct_prob, struct_prob + NUM_CLS));
        locs.push_back(loc);

        if (pre_char == EOS)
            break;
    }

    // ---- 4. 输出结果 ----
    // 4a. 把 token 序列转成 HTML 结构字符串, 同时收集 td 单元格的 bbox
    string html;
    vector<vector<float>> bbox_list;   // 每个 td 单元格的 8 个原图坐标 [x0,y0,...,x3,y3]
    for (size_t i = 0; i < tokens.size(); i++)
    {
        int t = tokens[i];
        if (t == 0 || t == EOS) continue;  // 跳过 sos / eos
        html += STRUCTURE_DICT[t];

        // 只有 <td (7) 和 <td></td> (48) 两个 token 有有效 bbox
        // (与 PaddleOCR TableLabelDecode 的 td_token = ["<td>", "<td", "<td></td>"] 一致)
        if (t == 7 || t == 48)
        {
            // 坐标还原 (PaddleOCR _bbox_decode):
            //   bbox[0::2] *= pad_w(488); bbox[1::2] *= pad_h(488)
            //   bbox[0::2] /= ratio;      bbox[1::2] /= ratio
            const vector<float>& loc = locs[i];
            vector<float> bbox(8);
            for (int j = 0; j < 8; j++)
            {
                // j%2==0 -> x 坐标, j%2==1 -> y 坐标
                float v = loc[j] * 488.0f / ratio;
                bbox[j] = v;
            }
            bbox_list.push_back(bbox);
        }
    }

    cout << "\n===== 解码结果 =====" << endl;
    cout << "tokens: ";
    for (size_t i = 0; i < tokens.size(); i++) cout << tokens[i] << " ";
    cout << endl;
    cout << "共 " << tokens.size() << " 步, 其中 " << bbox_list.size() << " 个单元格" << endl;

    // 4b. 输出带边框样式的完整 HTML
    cout << "\n===== 表格结构 (HTML) =====" << endl;
    cout << "<html><head><style>"
            "table{border-collapse:collapse;}"
            "td{border:1px solid #333;padding:4px 12px;text-align:center;min-width:20px;}"
            "</style></head><body><table>"
         << html
         << "</table></body></html>" << endl;

    // 4c. 单元格坐标输出
    cout << "\n===== 单元格坐标 (原图, 8 点 [x0,y0,...,x3,y3]) =====" << endl;
    for (size_t i = 0; i < bbox_list.size(); i++)
    {
        cout << "  cell " << i << ": ";
        for (int j = 0; j < 8; j++)
            cout << (j > 0 ? ", " : "") << (int)(bbox_list[i][j] + 0.5f);
        cout << endl;
    }

    // 4d. 可视化: 在原图上画框并保存结果图
    if (has_image && !bbox_list.empty())
    {
        cv::Mat vis = src_img.clone();
        for (size_t i = 0; i < bbox_list.size(); i++)
        {
            const vector<float>& b = bbox_list[i];
            // 8 点四边形 (左上->右上->右下->左下), 画 4 条边
            cv::Point pts[4];
            for (int k = 0; k < 4; k++)
                pts[k] = cv::Point((int)(b[k*2] + 0.5f), (int)(b[k*2+1] + 0.5f));
            cv::polylines(vis, std::vector<cv::Point>{pts[0], pts[1], pts[2], pts[3]},
                          true, cv::Scalar(0, 0, 255), 2);  // 红色边框
            // 标注序号
            cv::putText(vis, std::to_string(i), pts[0],
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        std::string out_path = "result_vis.jpg";
        cv::imwrite(out_path, vis);
        cout << "\n===== 可视化结果已保存: " << out_path << " =====" << endl;
    }

    // 4e. 逐 token 打印映射 (便于调试)
    cout << "\n===== token -> 字符映射 =====" << endl;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        int t = tokens[i];
        cout << "  [" << i << "] token=" << t << " -> \"" << STRUCTURE_DICT[t] << "\"" << endl;
    }

    return 0;
}
