	bits 64
	section .data
	
	extern g_valSet, g_index, g_pole_long, g_pole_long2
	extern g_char, g_index2, g_pozdrav
	extern g_long, g_lsb, g_msb

	section .text
	
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

	global setCharOnIndex
setCharOnIndex:
	enter 0,0

	MOV EAX, [g_index]
	MOV CL, [g_pozdrav + EAX + 1 ]
	MOV [g_pozdrav + EAX], CL 

	leave
	ret

	global setMsbLsb
setMsbLsb:
	enter 0,0

	MOV EAX, [g_long]
	MOV [g_lsb], EAX
	MOV EBX, [g_long+4]
	MOV [g_msb], EBX

	leave
	ret