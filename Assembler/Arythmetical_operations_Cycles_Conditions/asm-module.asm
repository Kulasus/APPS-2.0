	bits 64
	section .data
	
	extern g_long, g_pocet
	extern g_bin_num, g_bin_num_char
	extern g_date, g_pocet_cisel
	extern g_pocet4x0, g_pocet4x1, g_int_pole

	section .text

;-------------1)--------------------
	global spocitej_bity
spocitej_bity:
	enter 0,0

	mov DL, 0
	mov RCX, [g_long]

.back:
	cmp RCX, 0
	je .end
	shr RCX, 1
	jc .one
	jmp .back

.one:
	inc DL
	jmp .back

.end:
	mov [g_pocet], DL
	leave
	ret

;---------------5)--------------------------
	global num_to_bin_char
num_to_bin_char:
	enter 0,0

	mov RAX, 32
	mov RCX, [g_bin_num]

.back:
	cmp RCX, 0
	je .end
	shr RCX, 1
	jc .one
	mov byte [g_bin_num_char + RAX], '0'
	jmp .continue

.one:
	mov byte [g_bin_num_char + RAX], '1'
	

.continue:
	dec RAX
	jmp .back

.end:
	leave
	ret


;---------------4)--------------------------
	global spocitej_cisla
spocitej_cisla:
	enter 0,0

	mov RCX, 0
.back:
	cmp byte [g_date + RCX], 0
	je .endfor
	cmp byte [g_date + RCX], '0'
	je .count
	cmp byte [g_date + RCX], '1'
	je .count
	cmp byte [g_date + RCX], '2'
	je .count
	cmp byte [g_date + RCX], '3'
	je .count
	cmp byte [g_date + RCX], '4'
	je .count
	cmp byte [g_date + RCX], '5'
	je .count
	cmp byte [g_date + RCX], '6'
	je .count
	cmp byte [g_date + RCX], '7'
	je .count
	cmp byte [g_date + RCX], '8'
	je .count
	cmp byte [g_date + RCX], '9'
	je .count
	jmp .continue

.count:
	inc EAX

.continue:
	inc RCX
	jmp .back

.endfor:
	mov [g_pocet_cisel], EAX
	leave
	ret

;---------------2)--------------------------
	global spocitej_jednicky_nuly
spocitej_jednicky_nuly:
	enter 0,0

	
	mov EAX, 0
	mov EDX, 0
	mov EDI, 0

.back:
	cmp EAX, 255
	je .endfor
	mov ECX, [g_int_pole + EAX * 4]
	mov ESI, 15
	and ESI, ECX
	cmp ESI, 15
	je .ones
	cmp ESI, 0
	je .zeros
	jmp .continue

.ones:
	inc EDX
	jmp .continue
.zeros:
	inc EDI

.continue:
	inc EAX
	jmp .back

.endfor:
	mov [g_pocet4x0], EDI
	mov [g_pocet4x1], EDX
	leave
	ret
