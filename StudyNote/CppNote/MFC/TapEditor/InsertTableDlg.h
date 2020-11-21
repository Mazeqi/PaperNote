#if !defined(AFX_INSERTTABLEDLG_H__37322E00_57F9_45FC_9EB7_D1495517D818__INCLUDED_)
#define AFX_INSERTTABLEDLG_H__37322E00_57F9_45FC_9EB7_D1495517D818__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// InsertTableDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CInsertTableDlg dialog

class CInsertTableDlg : public CDialog
{
// Construction
public:
	CInsertTableDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CInsertTableDlg)
	enum { IDD = IDD_INSERT_TABLE };
	int		m_nColumns;
	int		m_nRows;
	//}}AFX_DATA


// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CInsertTableDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:

	// Generated message map functions
	//{{AFX_MSG(CInsertTableDlg)
		// NOTE: the ClassWizard will add member functions here
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_INSERTTABLEDLG_H__37322E00_57F9_45FC_9EB7_D1495517D818__INCLUDED_)
