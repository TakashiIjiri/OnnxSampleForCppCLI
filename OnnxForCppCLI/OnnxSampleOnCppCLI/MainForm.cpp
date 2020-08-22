#include "pch.h"
#include "MainForm.h"
#include <iostream>
#include "TCore.h"

using namespace OnnxSampleOnCppCLI;


MainForm::MainForm(void)
{
	InitializeComponent();
	m_b_mousedown = false;
	m_x = m_y = 0;
}



System::Void MainForm::m_pictbox_latent_MouseDown(
		System::Object^ sender, 
		System::Windows::Forms::MouseEventArgs^ e)
{
	m_b_mousedown = true;

}

System::Void MainForm::m_pictbox_latent_MouseUp(
		System::Object^ sender, 
		System::Windows::Forms::MouseEventArgs^ e)
{
	m_b_mousedown = false;

}

System::Void MainForm::m_pictbox_latent_MouseMove(
		System::Object^ sender, 
		System::Windows::Forms::MouseEventArgs^ e)
{
	if ( !m_b_mousedown ) return;
	m_x = (int)e->X;
	m_y = (int)e->Y;
	RedrawPictBoxes();
}



void MainForm::RedrawPictBoxes()
{
	//RUN onnx decoder 
	double latent_x = 5 * 2 * (m_x / (double)m_pictbox_latent->Width  - 0.5);
	double latent_y = 5 * 2 * (m_y / (double)m_pictbox_latent->Height - 0.5);
	unsigned char mnistimg[28*28];
	TCore::GetInst()->RunDecoder(latent_x, latent_y, mnistimg);
	

	//redraw pictbox (left)
	{
		const int W = m_pictbox_latent->Width;
		const int H = m_pictbox_latent->Height;

		Bitmap^ bmp = gcnew Bitmap(W, H);
		m_pictbox_latent->Image = bmp;
		Graphics^ graphics = Graphics::FromImage(m_pictbox_latent->Image);

		//draw frame
		Pen^ black_pen = gcnew Pen(Color::Black, 2);
		Pen^ red_pen = gcnew Pen(Color::Red, 2);
		graphics->DrawLine(black_pen, 0, (H - 1) / 2, W - 1, (H - 1) / 2);
		graphics->DrawLine(black_pen, (W - 1) / 2, H - 1, (W - 1) / 2, 0);
		graphics->DrawEllipse(red_pen, m_x, m_y, 10, 10);
		m_pictbox_latent->Refresh();
	}

	//redraw pictbox (right)
	{
	
		const int W = m_pictbox_output->Width;
		const int H = m_pictbox_output->Height;

		Bitmap^ bmp = gcnew Bitmap(W, H, Imaging::PixelFormat::Format24bppRgb);
		m_pictbox_output->Image = bmp;

		//bitmapを作成して，さらにpixelへのポインタも取得 
		System::Drawing::Rectangle rect = System::Drawing::Rectangle(0, 0, bmp->Width, bmp->Height);
		Imaging::BitmapData^ bmpData = bmp->LockBits(rect, Imaging::ImageLockMode::ReadWrite, bmp->PixelFormat);
		Byte* pBuf = (Byte*)bmpData->Scan0.ToPointer();

		for (int y = 0; y < H; ++y)
		{
			for (int x = 0; x < W; ++x)
			{
				int mx = (int)(x / (double)(W - 1) * (28 - 1) + 0.5);
				int my = (int)(y / (double)(H - 1) * (28 - 1) + 0.5);
				unsigned char c = mnistimg[mx + my*28];

				pBuf[y * bmpData->Stride + x * 3 + 0] = c;
				pBuf[y * bmpData->Stride + x * 3 + 1] = c;
				pBuf[y * bmpData->Stride + x * 3 + 2] = c;
			}
		}

		bmp->UnlockBits(bmpData);
		m_pictbox_output->Refresh();
	}
}

