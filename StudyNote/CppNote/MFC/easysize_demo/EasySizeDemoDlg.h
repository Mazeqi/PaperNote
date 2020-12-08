// EasySizeDemoDlg.h : header file
//

#if !defined(AFX_EASYSIZEDEMODLG_H__8E227342_30AC_44D9_B83A_52321DCD07CE__INCLUDED_)
#define AFX_EASYSIZEDEMODLG_H__8E227342_30AC_44D9_B83A_52321DCD07CE__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

/////////////////////////////////////////////////////////////////////////////
// CEasySizeDemoDlg dialog

class CEasySizeDemoDlg : public CDialog
{
DECLARE_EASYSIZE
// Construction
public:
	CEasySizeDemoDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	//{{AFX_DATA(CEasySizeDemoDlg)
	enum { IDD = IDD_EASYSIZEDEMO_DIALOG };
		// NOTE: the ClassWizard will add data members here
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CEasySizeDemoDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CEasySizeDemoDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnSizing(UINT fwSide, LPRECT pRect);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_EASYSIZEDEMODLG_H__8E227342_30AC_44D9_B83A_52321DCD07CE__INCLUDED_)
