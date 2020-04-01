bits 64

	section .data
	extern g_num

    section .text
; even_odd_count(int* array, int size, int* evenOdd);
; RDI, RSI, RDX, RCX, R8 a R9
	global even_odd_count
even_odd_count:
	enter 0,0

	push rbx
	mov eax,0 ; sude
	mov ecx,0 ; liche
	mov r11,0 ;i

.for:
	cmp r11, RSI
	je .endfor
	mov EBX, [RDI + R11 * 4]
	shr EBX, 1
	jc .odd

.even:
	inc eax
	jmp .back

.odd:
	inc ecx

.back:
	inc r11
	jmp .for


.endfor:
	mov [RDX], eax
	mov [RDX + 4], ecx

	pop rbx
	leave
	ret
;--------------------------------------------
;replace_chars(char* array, char replacer, char replaced)
; 					RDI, RSI, RDX, RCX, R8 a R9
	global replace_chars
replace_chars:
	enter 0,0

	push rbx
	mov RCX, 0
	mov AL, SIL
	sub AL, 32
	mov BL, DL
	sub BL, 32

.for:
	cmp byte[RDI + RCX], 0
	je .endfor

	cmp byte[RDI + RCX], SIL
	je .small
	cmp byte[RDI + RCX], AL
	je .big
	jmp .continue

.big:
	mov byte[RDI + RCX], BL
	jmp .continue

.small:
	mov byte[RDI + RCX], DL

.continue:
	inc RCX
	jmp .for

.endfor:	
	pop rbx
	leave
	ret

;--------------------------------------------
;str_to_num(char* array, long num)
; 					RDI, RSI, RDX, RCX, R8 a R9
	global str_to_num
str_to_num
	enter 0,0

	mov RCX,0
	mov RDX, 0
	mov RAX, 0

.for:
	cmp byte[RDI + RCX], 0
	je .endfor
	shl RAX, 1
	cmp byte[RDI + RCX], 49
	je .one
	jmp .continue
	
.one:
	inc RAX	

.continue:
	inc RCX
	jmp .for

.endfor:
	mov [g_num], RAX
	leave
	ret


	global test
test:
	enter 0,0
	mov EDI, 20
	leave
	ret