// TapBitmap.cpp: implementation of the CTapBitmap class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "TapEditor.h"
#include "TapBitmap.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CTapBitmap::CTapBitmap()
{
	m_pPicture	= NULL;
}

CTapBitmap::~CTapBitmap()
{
	if (m_pPicture != NULL)
	{
		m_pPicture->Release();
		m_pPicture = NULL;
	}
}

BOOL CTapBitmap::Load(CString sFileName)
{
	BOOL bResult = FALSE;

	if (m_pPicture != NULL)
	{
		m_pPicture->Release();
		m_pPicture = NULL;
	}

	if (m_pPicture == NULL)
	{
		CFile			cFile;
		CFileException	e;

		if (cFile.Open(sFileName, CFile::modeRead | CFile::typeBinary, &e))
		{
			BYTE* pBuff = new BYTE[cFile.GetLength()];

			if (cFile.Read(pBuff, cFile.GetLength()) > 0)
			{
				if (LoadFromBuffer(pBuff, cFile.GetLength()))
					bResult = TRUE;
			}

			delete [] pBuff;
		}
	}
	return bResult;
}

BOOL CTapBitmap::LoadFromBuffer(BYTE *pBuff, int nSize)
{
	BOOL bResult = FALSE;

	HGLOBAL hGlobal = GlobalAlloc(GMEM_MOVEABLE, nSize);
	void* pData = GlobalLock(hGlobal);
	memcpy(pData, pBuff, nSize);
	GlobalUnlock(hGlobal);

	IStream* pStream = NULL;

	if (CreateStreamOnHGlobal(hGlobal, TRUE, &pStream) == S_OK)
	{
		HRESULT hr;
		if ((hr = OleLoadPicture(pStream, nSize, FALSE, IID_IPicture, (LPVOID *)&m_pPicture)) == S_OK)
			bResult = TRUE;
		m_size=nSize;
		pStream->Release();
	}

	return bResult;
}

IPicture* CTapBitmap::GetBMP()
{
	return m_pPicture;
}