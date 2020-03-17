; 	This defines in which mode will nasm compile the code
	bits 64
; 	Data section is used for declaring variables
	section .data

; Variables
; 	extern means variables are not declared in this file
	extern g_char, g_num32, g_num64
	extern g_array_char, g_array_num32, g_array_num64
	extern g_index
	extern g_lastIndexOfArray
	extern g_firstIndexOfArray
	extern g_index_plus_one
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

;*****************************************************************
	; Definition of function switch between array num32 array num64 in asm
	global switch_between_array_num32_array_num64_in_asm
switch_between_array_num32_array_num64_in_asm:
	enter 0,0

	mov EAX, [g_array_num32 + 0 * 4]
	mov RBX, [g_array_num64 + 0 * 8]
	mov [g_array_num64 + 0 * 8], RAX
	mov [g_array_num32 + 0 * 4], EBX
	mov ECX, [g_array_num32 + 1 * 4]
	mov RDX, [g_array_num64 + 1 * 8]
	mov [g_array_num64 + 1 * 8], RCX
	mov [g_array_num32 + 1 * 4], EDX
	mov EDI, [g_array_num32 + 2 * 4]
	mov RSI, [g_array_num64 + 2 * 8]
	mov [g_array_num64 + 2 * 8], RDI
	mov [g_array_num32 + 2 * 4], ESI

	leave
	ret

;*****************************************************************
	; Definition of function switch first and last index array num32 in asm
	global switch_first_and_last_index_array_num32_in_asm
switch_first_and_last_index_array_num32_in_asm:
	enter 0,0

	mov ECX, [g_firstIndexOfArray]
	mov EDX, [g_lastIndexOfArray]
	mov EDI, [g_array_num32 + ECX * 4]
	mov ESI, [g_array_num32 + EDX * 4]
	mov [g_array_num32 + EDX * 4], EDI
	mov [g_array_num32 + ECX * 4], ESI

	leave
	ret

;*****************************************************************
	; Definition of function switch neighbors array num32 in asm
	global switch_neighbors_array_num32_in_asm
switch_neighbors_array_num32_in_asm:
	enter 0,0

	mov EAX, [g_array_num32 + 0 * 4]
	mov EBX, [g_array_num32 + 1 * 4]
	mov [g_array_num32 + 0 * 4], EBX
	mov [g_array_num32 + 1 * 4], EAX
	mov EAX, [g_array_num32 + 2 * 4]
	mov EBX, [g_array_num32 + 3 * 4]
	mov [g_array_num32 + 2 * 4], EBX
	mov [g_array_num32 + 3 * 4], EAX
	mov EAX, [g_array_num32 + 4 * 4]
	mov EBX, [g_array_num32 + 5 * 4]
	mov [g_array_num32 + 4 * 4], EBX
	mov [g_array_num32 + 5 * 4], EAX
	mov EAX, [g_array_num32 + 6 * 4]
	mov EBX, [g_array_num32 + 7 * 4]
	mov [g_array_num32 + 6 * 4], EBX
	mov [g_array_num32 + 7 * 4], EAX
	mov EAX, [g_array_num32 + 8 * 4]
	mov EBX, [g_array_num32 + 9 * 4]
	mov [g_array_num32 + 8 * 4], EBX
	mov [g_array_num32 + 9 * 4], EAX

	leave
	ret

;*****************************************************************
	; Definition of function set array num32 neighbors value
	; Does not work !!!
	global set_array_num32_neighbors_value
set_array_num32_neighbors_value:
	enter 0,0

	mov EAX, [g_index]
	mov ECX, [g_index_plus_one]
	mov EBX, [g_array_num32 + ECX * 4]
	mov [g_array_num32 + EAX * 4], EBX

	leave
	ret
; labels: