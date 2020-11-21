// TapBitmap.h: interface for the CTapBitmap class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _TAPBITMAP_H_
#define _TAPBITMAP_H_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CTapBitmap  
{
public:
	BOOL Load(CString sFileName);	
	IPicture* GetBMP();
	BOOL LoadFromBuffer(BYTE* pBuff, int nSize);

	CTapBitmap();
	virtual ~CTapBitmap();

private:
	IPicture*		m_pPicture;
	LONG			m_size;
};

#endif
