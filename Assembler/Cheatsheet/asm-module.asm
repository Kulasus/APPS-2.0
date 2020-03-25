	bits 64

	section .data

 extern g_val0, g_val1
 extern g_pole_long1, g_pole_long2, g_pozdrav
 extern g_index1
 extern g_long,g_lsb,g_msb

        section .text

	global uloha1
uloha1:
	enter 0,0

	
	movsx RCX, dword [g_val0]
	mov [g_pole_long1 + 0 * 8], RCX 
	movsx RCX, dword [g_val0]
	mov [g_pole_long1 + 2 * 8], RCX 
	movsx RCX, dword [g_val0]
	mov [g_pole_long1 + 4 * 8], RCX 
	movsx RCX, dword [g_val1] 
	mov [g_pole_long1 + 1 * 8], RCX 
	movsx RCX, dword [g_val1] 
	mov [g_pole_long1 + 3 * 8], RCX 
	movsx RCX, dword [g_val1] 
	mov [g_pole_long1 + 5 * 8], RCX 

	leave
	ret

	global uloha2
uloha2:
	enter 0,0

	mov RCX, qword [g_val0]
	mov [g_pole_long2 + 0 * 8], RCX 
	mov RCX, qword [g_val0]
	mov [g_pole_long2 + 2 * 8], RCX 
	mov RCX, qword [g_val0]
	mov [g_pole_long2 + 4 * 8], RCX 
	mov RCX, qword [g_val1] 
	mov [g_pole_long2 + 1 * 8], RCX 
	mov RCX, qword [g_val1] 
	mov [g_pole_long2 + 3 * 8], RCX 
	mov RCX, qword [g_val1] 
	mov [g_pole_long2 + 5 * 8], RCX 

	leave
	ret

	global uloha3
uloha3:
	enter 0,0

	mov BL, [g_pozdrav +1]
	mov [g_pozdrav + 0], BL
	mov BL, [g_pozdrav + 2]
	mov [g_pozdrav + 1], BL
	mov BL, [g_pozdrav + 3]
	mov [g_pozdrav + 2], BL
	mov BL, [g_pozdrav + 4]
	mov [g_pozdrav + 3], BL
	mov BL, [g_pozdrav + 5]
	mov [g_pozdrav + 4], BL
	mov BL, [g_pozdrav + 6]
	mov [g_pozdrav + 5], BL

	leave
	ret

	global uloha5
uloha5:
	enter 0,0

	mov EAX, [g_long]
    mov [g_lsb], EAX
	mov EAX, [g_long + 4]
	mov [g_msb], EAX

	leave
	ret


; labels:
