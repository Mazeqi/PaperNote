

#include "StdAfx.h"
#include "tapex.h"

long ClearString(CString str,char* cRes)
{
	if(cRes == NULL)
		return 0;

	long len = str.GetLength()*2;
	char* buf = new char[len + 1];
	memset(buf,0,len + 1);
	memcpy(buf,str.GetBuffer(len),len);
	CString sRes;
	long j = 0;
	for(LONG i = 0;i < len; i++)
	{
		if(buf[i] > 0)
		{
			cRes[j++] = buf[i];
		}
	}
	cRes[j] = 0;
	delete buf;
	return j;
}

void SetTextEX(HWND hWnd, CString csText, int nSTFlags, int nSTCodepage)
{
	SETTEXTEX st;
	st.codepage = nSTCodepage;	
	st.flags = nSTFlags;
	::SendMessage(hWnd, EM_SETTEXTEX, (WPARAM)&st, (LPARAM)(LPCTSTR)csText);
}