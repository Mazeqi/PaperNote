// EnBitmap.h: interface for the CEnBitmap class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _ENBITMAP_H_
#define _ENBITMAP_H_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CEnBitmap : public CBitmap  
{
public:
	CEnBitmap();
	virtual ~CEnBitmap();

	BOOL LoadImage(LPCTSTR szImagePath, COLORREF crBack = 0);
	BOOL LoadImage(UINT uIDRes, LPCTSTR szResourceType, HMODULE hInst = NULL, COLORREF crBack = 0); 

	// helpers
	static BOOL GetResource(LPCTSTR lpName, LPCTSTR lpType, HMODULE hInst, void* pResource, int& nBufSize);
	static IPicture* LoadFromBuffer(BYTE* pBuff, int nSize);
	BOOL Attach(IPicture* pPicture, COLORREF crBack);
protected:
	

};

#endif
