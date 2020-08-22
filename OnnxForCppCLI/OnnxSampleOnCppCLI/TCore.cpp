#include "pch.h"
#include "TCore.h"

#include <iostream>
#include <algorithm>

//note : 上のincludeを追加
//プロジェクトのプロパティよりonnxruntime.libファイルとそのディレクトリを指定するとうまくいく


#pragma unmanaged 


TCore::TCore()
{
	try {
		m_global_vae_mnist = std::make_unique<VAE_MNIST>();
	}
	catch (const Ort::Exception& exception) {
		std::cerr << exception.what() << std::endl;
		return;
	}

	//VAE MNISTの動作テスト
	//latent space について (-1,0) - (1,0) の間をサンプリングしてmnistを復元
	//標準出力に二値化結果を出力する
 
	for (int i = -10; i < 10; ++i)
	{
		m_global_vae_mnist->m_input = { i * 0.1f, 0 };
		auto res = m_global_vae_mnist->Run();

		std::cout << "input : " 
			<< m_global_vae_mnist->m_input[0] << " " 
			<< m_global_vae_mnist->m_input[1] << "\n";

		std::cout << "\n\noutput\n";
		for (int y = 0; y < 28; ++y)
		{
			for (int x = 0; x < 28; ++x)
				std::cout << (res[x + 28 * y] > 0.5 ? "@@" : "--");
			std::cout << "\n";
		}
	}

}


//VAE_MNISTにより潜在空間のベクトルから画像を生成する
void TCore::RunDecoder(double x, double y, unsigned char img[28 * 28])
{
	m_global_vae_mnist->m_input = { (float)x, (float)y};
	auto res = m_global_vae_mnist->Run();
	for ( int i=0; i < 28*28; ++i) 
	{
		float c = std::max( 0.0f, std::min(res[i], 1.0f) );
		img[i] = (unsigned char) (255 * c);
	}
}

#pragma managed
