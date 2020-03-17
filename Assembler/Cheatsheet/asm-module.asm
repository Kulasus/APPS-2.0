; 	This defines in which mode will nasm compile the code
	bits 64
; 	Data section is used for declaring variables
	section .data

; Variables
; 	extern means variables are not declared in this file
	extern g_char, g_num32, g_num64
	extern g_array_char, g_array_num32, g_array_num64
	extern g_index
; Text section is used for defining functions
	section .text
;*****************************************************************
	; Definition of function set_variables_in_asm
	global set_variables_in_asm
set_variables_in_asm:
	enter 0,0 ; Equiualent to {
	
	; Assigning variables value from memory to memory
	mov byte [g_char], 'x' 
	mov dword [g_num32], 32323232
	mov qword [g_num64], 64646464

	leave	; Equiualent to }
	ret

;*****************************************************************
	; Definition of function move_variables_in_asm
	global move_variables_in_asm
move_variables_in_asm:
	enter 0,0

	; Extension of g_num32 by movsx because g_num32 is 32bit and RCX is 64bit
	movsx RCX, dword[g_num32]
	mov [g_num64], RCX

	; The same, only difference here is we are extending g_char only to 32bit
	movsx EAX, byte[g_char]
	mov [g_num32], EAX

	leave
	ret

;*****************************************************************
	; Definition of function set_array_char_in_asm
	global set_array_char_in_asm
set_array_char_in_asm:
	enter 0,0

	mov byte [g_array_char], 'K'
	mov byte [g_array_char + 1], 'O'
	mov byte [g_array_char + 2], 'K'
	mov byte [g_array_char + 3], 'O'

	leave
	ret

;*****************************************************************
	; Definition of function set_array_num64_in_asm
	global set_array_num64_in_asm
set_array_num64_in_asm:
	enter 0,0

	; 1 * 8 because + 1 increases adress only by 8bits
	mov qword [g_array_num64 + 0 * 8], 1000
	mov qword [g_array_num64 + 4 * 8], 1000
	mov qword [g_array_num64 + 9 * 8], 1000

	leave
	ret

;*****************************************************************
	; Definition of function set_array_num32_index_in_asm
	global set_array_num32_index_in_asm
set_array_num32_index_in_asm:
	enter 0,0

	movsx RAX, dword [g_index]
	mov dword [g_array_num32 + RAX * 4 ], 3333

	leave
	ret

;*****************************************************************
	; Definition of function set_array_num32_index_num32_in_asm
	global set_array_num32_index_num32_in_asm
set_array_num32_index_num32_in_asm:
	enter 0,0

	mov EAX, [g_index]
	mov EBX, [g_num32]
	mov [g_array_num32 + EAX * 4], EBX

	leave
	ret

; labels:
