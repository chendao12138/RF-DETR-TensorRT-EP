#include "inference.h"
#include <regex>

#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))

#define benchmark
DETR::DETR() {

}


DETR::~DETR() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}

static inline float clampf(float v, float lo, float hi) {
    return max(lo, min(v, hi));
}

static inline void softmax_stable(const float* logits, int n, std::vector<float>& probs) {
    probs.resize(n);
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i) m = max(m, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::exp((double)logits[i] - (double)m);
        probs[i] = (float)e;
        sum += e;
    }
    float inv = (sum > 0) ? (float)(1.0 / sum) : (1.0f / n);
    for (int i = 0; i < n; ++i) probs[i] *= inv;
}

char* DETR::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }


    if (iImg.cols >= iImg.rows)
    {
        resizeScales = iImg.cols / (float)iImgSize.at(0);
        cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
    }
    else
    {
        resizeScales = iImg.rows / (float)iImgSize.at(0);
        cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
    }
    cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
    oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
    oImg = tempImg;

    return RET_OK;
}


char* DETR::CreateSession(INIT_PARAMs& iParams) {
    char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        char output[] = "[DETR]:Your model path is error.Change your model path without chinese characters.";
        Ret = output;
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
		detThreshold = iParams.detThreshold;
        imgSize = iParams.imgSize;
        cudaEnable = iParams.cudaEnable;
        trtEnable = iParams.trtEnable;
		modelPath = iParams.modelPath;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Detr");
        Ort::SessionOptions sessionOption;
        if (trtEnable)
        {
            const auto& api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;
            api.CreateTensorRTProviderOptions(&tensorrt_options);
            std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
                tensorrt_options, api.ReleaseTensorRTProviderOptions);
            std::vector<const char*> keys{ "device_id", "trt_fp16_enable", "trt_int8_enable", "trt_engine_cache_enable","trt_engine_cache_path" };
            std::vector<const char*> values{ "0", "1", "0", "1","./trt_engine_cache_path" };
            api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), keys.size());
            api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(sessionOption),
                rel_trt_options.get());
        }
        else if (cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), static_cast<int>(modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), static_cast<int>(modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy_s(temp_buf, 50, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy_s(temp_buf, 50, output_node_name.get());
			std::cout << "Output node name: " << temp_buf << std::endl;
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[DETR]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        strcpy_s(merged, result.length() + 1, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;

        char output[] = "[DETR]:Create session failed.";
        Ret = output;
        return Ret;
    }

}


char* DETR::RunSession(cv::Mat& iImg, std::vector<Det>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif // benchmark

    char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    float* blob = new float[processedImg.total() * 3];
    BlobFromImage(processedImg, blob);
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);

    return Ret;
}

std::vector<Det> DETR::postprocess(
    const float* dets_b,      // [Q,4] cxcywh (0..1)
    const float* logits_b,    // [Q,num_classes] logits
    int64_t Q
) {
    const int C = this->classes.size();
    std::vector<Det> out;
    out.reserve((size_t)Q);

    std::vector<float> probs;
    for (int64_t q = 0; q < Q; ++q) {
        const float* box = dets_b + q * 4;
        const float* logit = logits_b + q * C;

        softmax_stable(logit, C, probs);

        int cls = 0;
        float score = probs[0];
        for (int c = 1; c < C; ++c) {
            if (probs[c] > score) { score = probs[c]; cls = c; }
        }

        //if (cls == background_id) continue;
        if (score < detThreshold) continue;

        float x = box[0] * imgSize[0];
        float y = box[1] * imgSize[1];
        float w = box[2] * imgSize[0];
        float h = box[3] * imgSize[1];
        int left = int((x - 0.5 * w) * resizeScales);
        int top = int((y - 0.5 * h) * resizeScales);

        int width = int(w * resizeScales);
        int height = int(h * resizeScales);

        Det d;
		d.box = cv::Rect(left, top, width, height);
        d.confidence = score;
        d.classId = cls;
        out.push_back(d);
    }
    return out;
}


template<typename N>
char* DETR::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<Det>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif // benchmark
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif // benchmark

    const int num_classes = this->classes.size();

    Ort::Value& detsTensor = outputTensor[0];   // expect [B,Q,4] or [Q,4]
    Ort::Value& labelsTensor = outputTensor[1]; // expect [B,Q,num_classes] or [Q,num_classes]

    auto detsInfo = detsTensor.GetTensorTypeAndShapeInfo();
    auto labInfo = labelsTensor.GetTensorTypeAndShapeInfo();

    auto detsDims = detsInfo.GetShape();
    auto labDims = labInfo.GetShape();

	// simple checks
    if (detsDims.empty() || labDims.empty()) {
        throw std::runtime_error("Empty output shape.");
    }
    if (detsDims.back() != 4) {
        throw std::runtime_error("dets last dim is not 4 (expected cxcywh).");
    }

    int64_t B = 1, Q = 0;

    if (detsDims.size() == 3) {
        B = detsDims[0];
        Q = detsDims[1];
    }
    else if (detsDims.size() == 2) {
        B = 1;
        Q = detsDims[0];
    }
    else {
        throw std::runtime_error("Unexpected dets dims rank.");
    }

    int64_t B2 = 1, Q2 = 0;
    if (labDims.size() == 3) {
        B2 = labDims[0];
        Q2 = labDims[1];
    }
    else if (labDims.size() == 2) {
        B2 = 1;
        Q2 = labDims[0];
    }
    else {
        throw std::runtime_error("Unexpected labels dims rank.");
    }

    if (B != B2 || Q != Q2) {
        throw std::runtime_error("dets and labels shapes mismatch.");
    }

    const float* dets_ptr = detsTensor.GetTensorData<float>();   // [B,Q,4]
    const float* logits_ptr = labelsTensor.GetTensorData<float>(); // [B,Q,num_classes]

    //expect batch size = 1, if not, only process the first batch, or for (int b = 0; b < B; ++b)
    int b = 0;  
    const float* dets_b = dets_ptr + b * (Q * 4);
    const float* logits_b = logits_ptr + b * (Q * num_classes);
    auto dets = postprocess(dets_b, logits_b, Q);

    oResult = dets;
#ifdef benchmark
    clock_t starttime_4 = clock();
    double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
    double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
    double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
    if (cudaEnable)
    {
        std::cout << "[DETR(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
    }
    else if (trtEnable)
    {
        std::cout << "[DETR(TRT)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
    }
    else
    {
        std::cout << "[DETR(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
    }
#endif // benchmark
    return RET_OK;

}


char* DETR::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    float* blob = new float[iImg.total() * 3];
    BlobFromImage(processedImg, blob);
    std::vector<int64_t> DETR_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        DETR_input_node_dims.data(), DETR_input_node_dims.size());
    auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    delete[] blob;
    clock_t starttime_4 = clock();
    double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
    if (cudaEnable)
    {
        std::cout << "[DETR(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
    }
    return RET_OK;
}
