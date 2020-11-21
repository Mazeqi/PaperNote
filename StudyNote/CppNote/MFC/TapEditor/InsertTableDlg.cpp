// InsertTableDlg.cpp : implementation file
//

#include "stdafx.h"
#include "TapEditor.h"
#include "InsertTableDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CInsertTableDlg dialog


CInsertTableDlg::CInsertTableDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CInsertTableDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CInsertTableDlg)
	m_nColumns = 2;
	m_nRows = 2;
	//}}AFX_DATA_INIT
}


void CInsertTableDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CInsertTableDlg)
	DDX_Text(pDX, IDC_NUM_COLUMNS, m_nColumns);
	DDV_MinMaxInt(pDX, m_nColumns, 1, 200);
	DDX_Text(pDX, IDC_NUM_ROWS, m_nRows);
	DDV_MinMaxInt(pDX, m_nRows, 1, 200);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CInsertTableDlg, CDialog)
	//{{AFX_MSG_MAP(CInsertTableDlg)
		// NOTE: the ClassWizard will add message map macros here
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CInsertTableDlg message handlers
