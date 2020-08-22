#pragma once

namespace OnnxSampleOnCppCLI {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// MainForm の概要
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	private:
		MainForm(void);
		bool m_b_mousedown;
		int  m_x, m_y;
		void RedrawPictBoxes();


	public: 
		static MainForm^ m_singleton;
	public:
		static MainForm^ GetInst()
		{
			if (m_singleton == nullptr) m_singleton = gcnew MainForm();
			return m_singleton;
		}
		
	protected:
		/// <summary>
		/// 使用中のリソースをすべてクリーンアップします。
		/// </summary>
		~MainForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::PictureBox^ m_pictbox_latent;
	private: System::Windows::Forms::PictureBox^ m_pictbox_output;
	protected:

	protected:


	private:
		/// <summary>
		/// 必要なデザイナー変数です。
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// デザイナー サポートに必要なメソッドです。このメソッドの内容を
		/// コード エディターで変更しないでください。
		/// </summary>
		void InitializeComponent(void)
		{
			this->m_pictbox_latent = (gcnew System::Windows::Forms::PictureBox());
			this->m_pictbox_output = (gcnew System::Windows::Forms::PictureBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->m_pictbox_latent))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->m_pictbox_output))->BeginInit();
			this->SuspendLayout();
			// 
			// m_pictbox_latent
			// 
			this->m_pictbox_latent->BackColor = System::Drawing::SystemColors::ButtonHighlight;
			this->m_pictbox_latent->Location = System::Drawing::Point(12, 12);
			this->m_pictbox_latent->Name = L"m_pictbox_latent";
			this->m_pictbox_latent->Size = System::Drawing::Size(284, 264);
			this->m_pictbox_latent->TabIndex = 0;
			this->m_pictbox_latent->TabStop = false;
			this->m_pictbox_latent->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::m_pictbox_latent_MouseDown);
			this->m_pictbox_latent->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::m_pictbox_latent_MouseMove);
			this->m_pictbox_latent->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::m_pictbox_latent_MouseUp);
			// 
			// m_pictbox_output
			// 
			this->m_pictbox_output->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->m_pictbox_output->Location = System::Drawing::Point(302, 54);
			this->m_pictbox_output->Name = L"m_pictbox_output";
			this->m_pictbox_output->Size = System::Drawing::Size(190, 181);
			this->m_pictbox_output->TabIndex = 1;
			this->m_pictbox_output->TabStop = false;
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 12);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(500, 284);
			this->Controls->Add(this->m_pictbox_output);
			this->Controls->Add(this->m_pictbox_latent);
			this->Name = L"MainForm";
			this->Text = L"MainForm";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->m_pictbox_latent))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->m_pictbox_output))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void m_pictbox_latent_MouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
	private: System::Void m_pictbox_latent_MouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
	private: System::Void m_pictbox_latent_MouseMove(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e);
	};
}
