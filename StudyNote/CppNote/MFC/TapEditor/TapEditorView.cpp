// TapEditorView.cpp : implementation of the CTapEditorView class
//

#include "stdafx.h"
#include "TapEditor.h"

#include "TapEditorDoc.h"
#include "CntrItem.h"
#include "TapEditorView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

static const TCHAR szClassRE[] = TEXT("RICHEDIT50W");

/////////////////////////////////////////////////////////////////////////////
// CTapEditorView

IMPLEMENT_DYNCREATE(CTapEditorView, CRichEditView)

BEGIN_MESSAGE_MAP(CTapEditorView, CRichEditView)
	//{{AFX_MSG_MAP(CTapEditorView)
	ON_WM_DESTROY()
	ON_COMMAND(ID_INSERT_TABLE, OnInsertTable)
	ON_COMMAND(ID_INSERT_IMAGE, OnInsertImage)
	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CRichEditView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CRichEditView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CRichEditView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CTapEditorView construction/destruction

CTapEditorView::CTapEditorView()
{
	// TODO: add construction code here

}

CTapEditorView::~CTapEditorView()
{
}

BOOL CTapEditorView::PreCreateWindow(CREATESTRUCT& cs)
{
	BOOL bRes = CRichEditView::PreCreateWindow(cs);
	cs.style |= ES_SELECTIONBAR;
	cs.lpszClass = szClassRE;
	return bRes;
}

void CTapEditorView::OnInitialUpdate()
{
	CRichEditView::OnInitialUpdate();


	// Set the printing margins (720 twips = 1/2 inch).
	SetMargins(CRect(720, 720, 720, 720));
}

/////////////////////////////////////////////////////////////////////////////
// CTapEditorView printing

BOOL CTapEditorView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}


void CTapEditorView::OnDestroy()
{
	// Deactivate the item on destruction; this is important
	// when a splitter view is being used.
   CRichEditView::OnDestroy();
   COleClientItem* pActiveItem = GetDocument()->GetInPlaceActiveItem(this);
   if (pActiveItem != NULL && pActiveItem->GetActiveView() == this)
   {
      pActiveItem->Deactivate();
      ASSERT(GetDocument()->GetInPlaceActiveItem(this) == NULL);
   }
}


/////////////////////////////////////////////////////////////////////////////
// CTapEditorView diagnostics

#ifdef _DEBUG
void CTapEditorView::AssertValid() const
{
	CRichEditView::AssertValid();
}

void CTapEditorView::Dump(CDumpContext& dc) const
{
	CRichEditView::Dump(dc);
}

CTapEditorDoc* CTapEditorView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CTapEditorDoc)));
	return (CTapEditorDoc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CTapEditorView message handlers

#include "tapex.h"
#include "InsertTableDlg.h"

void CTapEditorView::OnInsertTable() 
{
	CInsertTableDlg dlg;
	
	if(dlg.DoModal() == IDCANCEL)
		return;

	int rows = dlg.m_nRows,
		cols = dlg.m_nColumns;
	
	CString s = "{\\rtf1";
	CString sTable = s + "\\ansi\\ansicpg1252\\deff0\\deflang1033{\\fonttbl{\\f0\\froman\\fprq2\\fcharset0 Times New Roman;}{\\f1\\fswiss\\fcharset0 Arial;}}{\\*\\generator Msftedit 5.41.15.1503;}\\viewkind4\\uc1";
	CString row,col;
	row = "\\trowd\\trgaph108\\trleft8\\trbrdrl\\brdrs\\brdrw10 \\trbrdrt\\brdrs\\brdrw10 \\trbrdrr\\brdrs\\brdrw10 \\trbrdrb\\brdrs\\brdrw10 \\trpaddl108\\trpaddr108\\trpaddfl3\\trpaddfr3";
	col = "\\clbrdrl\\brdrw10\\brdrs\\clbrdrt\\brdrw10\\brdrs\\clbrdrr\\brdrw10\\brdrs\\clbrdrb\\brdrw10\\brdrs\\cellx";
	CString endcell = "\\cell";
	CString endrow = "\\row";
	int i,j;
	int width = 8748/cols;
	CString sColw;
	for(i=0;i<rows;i++)
	{
		sTable += row;
		for(j=0;j<cols;j++)
		{
			sTable += col;
			sColw.Format(_T("%d"),width *(j+1));
			sTable += sColw;			 
		}
		sTable += "\\pard\\intbl";
		for(j=0;j<cols;j++)
		{
			sTable += endcell;
		}
		sTable += endrow;
	}
	sTable += "\\par}";
	
#ifdef _UNICODE
	LONG len = sTable.GetLength() * 2;
	char* data = new char[len + 1];
	ClearString(sTable, data);
	SETTEXTEX st;
	st.codepage = 1200;	
	st.flags = ST_SELECTION | ST_KEEPUNDO;
	SendMessage(EM_SETTEXTEX, (WPARAM)&st, (LPARAM)(LPCTSTR)data);
	delete data;
#else
	SetTextEX(m_hWnd, sTable, ST_SELECTION|ST_KEEPUNDO, 1200);
#endif
	
}

#include "TapBitmap.h"
#include "EnBitmap.h"
#include "ImageDataObject.h"

void CTapEditorView::OnInsertImage() 
{
	// TODO: Add your command handler code here
	CString sFilter = "All image file|*.bmp;*.jpg;*.gif|Bitmap Files (*.bmp)|*.bmp|JPEG Files (*.jpg)|*.jpg|GIF Files (*.gif)|*.gif|";
	CFileDialog dlg(TRUE, NULL, NULL, OFN_FILEMUSTEXIST|OFN_READONLY, sFilter);
	if(dlg.DoModal() == IDOK)
	{
		CTapBitmap bmp;
		if(bmp.Load(dlg.GetPathName())==FALSE)
		{
			AfxMessageBox(_T("Could not load image."));
			return;
		}
		CEnBitmap enBitmap;		
		CBitmap Bitmap;
		if (enBitmap.Attach(bmp.GetBMP(), 0))
		{
			Bitmap.DeleteObject();
			Bitmap.Attach(enBitmap.Detach());
						
			IRichEditOle	*pRichEditOle;
			pRichEditOle = GetRichEditCtrl().GetIRichEditOle();
			HBITMAP hBitmap = (HBITMAP)Bitmap;
			if(hBitmap)
			{
				CImageDataObject::InsertBitmap(pRichEditOle, hBitmap);
			}
		}
	}
}
