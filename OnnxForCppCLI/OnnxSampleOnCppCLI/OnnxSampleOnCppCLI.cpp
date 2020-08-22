#include "pch.h"

#include <iostream>
#include "TCore.h"
#include "MainForm.h"


using namespace std;
using namespace System;
using namespace OnnxSampleOnCppCLI;

int main()
{
  
  cout << "Hello, World!\n";
  TCore::GetInst();

  cout << "FormMain::getInst()->ShowDialog() \n";
  MainForm::GetInst()->ShowDialog();

  return 0;
}
