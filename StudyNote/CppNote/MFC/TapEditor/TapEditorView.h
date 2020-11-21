// TapEditorView.h : interface of the CTapEditorView class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_TAPEDITORVIEW_H__F05122CC_5AF4_4EE2_84B7_26CD8B256E18__INCLUDED_)
#define AFX_TAPEDITORVIEW_H__F05122CC_5AF4_4EE2_84B7_26CD8B256E18__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CTapEditorCntrItem;

class CTapEditorView : public CRichEditView
{
protected: // create from serialization only
	CTapEditorView();
	DECLARE_DYNCREATE(CTapEditorView)

// Attributes
public:
	CTapEditorDoc* GetDocument();

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CTapEditorView)
	public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual void OnInitialUpdate(); // called first time after construct
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CTapEditorView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CTapEditorView)
	afx_msg void OnDestroy();
	afx_msg void OnInsertTable();
	afx_msg void OnInsertImage();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in TapEditorView.cpp
inline CTapEditorDoc* CTapEditorView::GetDocument()
   { return (CTapEditorDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TAPEDITORVIEW_H__F05122CC_5AF4_4EE2_84B7_26CD8B256E18__INCLUDED_)
