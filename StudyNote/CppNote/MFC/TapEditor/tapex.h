

#ifndef _TAPEX_H_
#define _TAPEX_H_

#define EM_SETTEXTEX			(WM_USER + 97)

// Flags for the SETEXTEX data structure 
#define ST_DEFAULT		0
#define ST_KEEPUNDO		1
#define ST_SELECTION	2
#define ST_NEWCHARS 	4

// Flags for the GETEXTEX data structure 
#define GT_DEFAULT		0
#define GT_USECRLF		1
#define GT_SELECTION	2
#define GT_RAWTEXT		4
#define GT_NOHIDDENTEXT	8

// EM_SETTEXTEX info; this struct is passed in the wparam of the message 
typedef struct _settextex
{
	DWORD	flags;			// Flags (see the ST_XXX defines)			
	UINT	codepage;		// Code page for translation (CP_ACP for sys default,
						    //  1200 for Unicode, -1 for control default)	
} SETTEXTEX;

long ClearString(CString str,char* cRes);
void SetTextEX(HWND hWnd, CString csText, int nSTFlags, int nSTCodepage);

#endif