	bits 64
	section .data
	
	extern g_val0, g_val1, g_valSet, g_index, g_pole_long, g_pole_long2
	extern g_char, g_index2, g_pozdrav
	extern g_long, g_lsb, g_msb

	section .text
;new-----------------------------
	global setLongArray
setLongArray:
	enter 0,0
	
	MOVSX RAX, dword [g_val0]
	MOVSX RBX, dword [g_val1]
	MOV [g_pole_long], RAX
	MOV [g_pole_long + 1 * 8], RBX
	MOV [g_pole_long + 2 * 8], RAX
	MOV [g_pole_long + 3 * 8], RBX
	MOV [g_pole_long + 4 * 8], RAX
	MOV [g_pole_long + 5 * 8], RBX

	leave
	ret

	global setLongArrayNoExtension
setLongArrayNoExtension:
	enter 0,0
	
	MOV RAX, qword [g_val0]
	MOV RCX, qword [g_val1]
	MOV [g_pole_long2 + 0 * 8], RAX
	MOV [g_pole_long2 + 1 * 8], RCX
	MOV [g_pole_long2 + 2 * 8], RAX
	MOV [g_pole_long2 + 3 * 8], RCX
	MOV [g_pole_long2 + 4 * 8], RAX
	MOV [g_pole_long2 + 5 * 8], RCX

	leave
	ret
;old-----------------------------
	global setLongOnIndex
setLongOnIndex:
	enter 0,0
	
	MOV EAX, [g_index]
	MOVSX RBX, dword [g_valSet]
	MOV [g_pole_long + EAX * 8], RBX

	leave
	ret

	global setLongOnIndexNoExtension
setLongOnIndexNoExtension:
	enter 0,0

	MOV EAX, [g_index]
	MOV RBX, qword [g_valSet]
	MOV [g_pole_long2 + EAX * 8], RBX

	leave
	ret
;new-----------------------------
	global moveCharArrayLeft
moveCharArrayLeft:
	enter 0,0

	MOV CL, [g_pozdrav+1]
	MOV [g_pozdrav], CL
	MOV CL, [g_pozdrav+2]
	MOV [g_pozdrav + 1], CL
	MOV CL, [g_pozdrav+3]
	MOV [g_pozdrav + 2], CL
	MOV CL, [g_pozdrav+4]
	MOV [g_pozdrav + 3], CL
	MOV CL, [g_pozdrav+5]
	MOV [g_pozdrav + 4], CL
	MOV CL, [g_pozdrav+6]
	MOV [g_pozdrav + 5], CL

	leave
	ret
;old-----------------------------
	global setCharOnIndex
setCharOnIndex:
	enter 0,0

	MOV EAX, [g_index]
	MOV CL, [g_pozdrav + EAX + 1 ]
	MOV [g_pozdrav + EAX], CL 

	leave
	ret
;--------------------------------
	global setMsbLsb
setMsbLsb:
	enter 0,0

	MOV EAX, [g_long]
	MOV [g_lsb], EAX
	MOV EBX, [g_long+4]
	MOV [g_msb], EBX

	leave
	ret