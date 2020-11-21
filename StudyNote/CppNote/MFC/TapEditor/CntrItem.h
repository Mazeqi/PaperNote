// CntrItem.h : interface of the CTapEditorCntrItem class
//

#if !defined(AFX_CNTRITEM_H__A7E81491_1AF9_4673_B917_71822D815503__INCLUDED_)
#define AFX_CNTRITEM_H__A7E81491_1AF9_4673_B917_71822D815503__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CTapEditorDoc;
class CTapEditorView;

class CTapEditorCntrItem : public CRichEditCntrItem
{
	DECLARE_SERIAL(CTapEditorCntrItem)

// Constructors
public:
	CTapEditorCntrItem(REOBJECT* preo = NULL, CTapEditorDoc* pContainer = NULL);
		// Note: pContainer is allowed to be NULL to enable IMPLEMENT_SERIALIZE.
		//  IMPLEMENT_SERIALIZE requires the class have a constructor with
		//  zero arguments.  Normally, OLE items are constructed with a
		//  non-NULL document pointer.

// Attributes
public:
	CTapEditorDoc* GetDocument()
		{ return (CTapEditorDoc*)CRichEditCntrItem::GetDocument(); }
	CTapEditorView* GetActiveView()
		{ return (CTapEditorView*)CRichEditCntrItem::GetActiveView(); }

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CTapEditorCntrItem)
	public:
	protected:
	//}}AFX_VIRTUAL

// Implementation
public:
	~CTapEditorCntrItem();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CNTRITEM_H__A7E81491_1AF9_4673_B917_71822D815503__INCLUDED_)
