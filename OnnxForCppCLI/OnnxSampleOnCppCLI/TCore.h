#pragma once

#pragma unmanaged

#include "onnxruntime_cxx_api.h"


//https://tadaoyamaoka.hatenablog.com/entry/2020/05/26/233159

struct VAE_MNIST
{
private:
	// これが複数あってはだめ見たい．
	//これが一つでセッションが複数はOKみたい（未確認）．
	Ort::Env m_env;

	Ort::Session m_session{ m_env, L"decoder.onnx", Ort::SessionOptions{nullptr} };

	Ort::Value m_input_tensor{ nullptr };
	Ort::Value m_output_tensor{ nullptr };

	std::array<int64_t, 2> m_input_shape{ 1, 2 };
	std::array<int64_t, 2> m_output_shape{ 1, 28 * 28 };

public:
	std::array<float, 2> m_input{};
	std::array<float, 28 * 28> m_output{};
	int64_t m_result{ 0 };


	VAE_MNIST()
	{
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		m_input_tensor = Ort::Value::CreateTensor<float>(
				memory_info,
				m_input.data(), m_input.size(), 
				m_input_shape.data(), m_input_shape.size());
		m_output_tensor = Ort::Value::CreateTensor<float>(
				memory_info,
				m_output.data(), m_output.size(), 
				m_output_shape.data(), m_output_shape.size());
	}

	std::array<float, 28 * 28> Run()
	{
		//この名前は onnxファイルをnutronというソフトで読み込むと確認できる
		const char* input_names[] = { "input.1" };
		const char* output_names[] = { "12" };
		m_session.Run(Ort::RunOptions{ nullptr }, input_names, &m_input_tensor, 1,
			output_names, &m_output_tensor, 1);
		return m_output;
	}
};



class TCore
{
private: 
	TCore();
	std::unique_ptr<VAE_MNIST> m_global_vae_mnist;

public:
  static TCore* GetInst() {
    static TCore p;
    return &p;
  }

	void RunDecoder(double x, double y, unsigned char img[28 * 28]);
};



#pragma managed
