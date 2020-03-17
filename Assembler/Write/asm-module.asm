	bits 64
	section .data
; Variables
	extern g_message ; Message to be writen

	section .text
;*****************************************************************
	; Definition of function set_message
	global set_message
set_message:
	enter 0,0

	mov byte [g_message], 'S'
	mov byte [g_message + 1], 'y'
	mov byte [g_message + 2], 's'
	mov byte [g_message + 3], 't'
	mov byte [g_message + 4], 'e'
	mov byte [g_message + 5], 'm'
	mov byte [g_message + 6], ' '
	mov byte [g_message + 7], 'f'
	mov byte [g_message + 8], 'u'
	mov byte [g_message + 9], 'n'
	mov byte [g_message + 10], 'g'
	mov byte [g_message + 11], 'u'
	mov byte [g_message + 12], 'j'
	mov byte [g_message + 13], 'e'
	mov byte [g_message + 14], '.'

	leave
	ret
