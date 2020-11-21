// CntrItem.cpp : implementation of the CTapEditorCntrItem class
//

#include "stdafx.h"
#include "TapEditor.h"

#include "TapEditorDoc.h"
#include "TapEditorView.h"
#include "CntrItem.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CTapEditorCntrItem implementation

IMPLEMENT_SERIAL(CTapEditorCntrItem, CRichEditCntrItem, 0)

CTapEditorCntrItem::CTapEditorCntrItem(REOBJECT* preo, CTapEditorDoc* pContainer)
	: CRichEditCntrItem(preo, pContainer)
{
	// TODO: add one-time construction code here
	
}

CTapEditorCntrItem::~CTapEditorCntrItem()
{
	// TODO: add cleanup code here
	
}

/////////////////////////////////////////////////////////////////////////////
// CTapEditorCntrItem diagnostics

#ifdef _DEBUG
void CTapEditorCntrItem::AssertValid() const
{
	CRichEditCntrItem::AssertValid();
}

void CTapEditorCntrItem::Dump(CDumpContext& dc) const
{
	CRichEditCntrItem::Dump(dc);
}
#endif

/////////////////////////////////////////////////////////////////////////////
