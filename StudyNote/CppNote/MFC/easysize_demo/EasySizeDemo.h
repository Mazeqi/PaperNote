// EasySizeDemo.h : main header file for the EASYSIZEDEMO application
//

#if !defined(AFX_EASYSIZEDEMO_H__D3C265E9_B9D5_4E5C_8413_4FCB666C1776__INCLUDED_)
#define AFX_EASYSIZEDEMO_H__D3C265E9_B9D5_4E5C_8413_4FCB666C1776__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols

/////////////////////////////////////////////////////////////////////////////
// CEasySizeDemoApp:
// See EasySizeDemo.cpp for the implementation of this class
//

class CEasySizeDemoApp : public CWinApp
{
public:
	CEasySizeDemoApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CEasySizeDemoApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation

	//{{AFX_MSG(CEasySizeDemoApp)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_EASYSIZEDEMO_H__D3C265E9_B9D5_4E5C_8413_4FCB666C1776__INCLUDED_)
