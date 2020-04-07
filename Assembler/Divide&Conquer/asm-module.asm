	bits 64

	section .data

	section .text 

	global vyskyt
vyskyt:
	enter 1024,0 ;intove pole o 256 pozicich | 256 * 4 = 1024

	mov RCX, 256
.while:
	dec RCX
	mov [RBP - 1024 + RCX * 4], dword 0
	jnz .while

	mov RAX, 0 ;ret
	mov RDX, 0 ;max
	mov RCX, 0 ;i

.back:
	cmp byte [RDI + RCX], 0
	je .done
	
	movzx RSI, byte [RDI + RCX] ;index = str[i]
	inc dword[RBP - 1024 + RSI * 4] ;citace

	cmp EDX, [RBP - 1024 + RSI * 4]
	cmovl EDX, [RBP - 1024 + RSI * 4]
	cmovl RAX, RSI

	inc RCX
	jmp .back

.done:
	leave
	ret




	global faktorial
faktorial:
	enter 16,0

	mov RAX, 1
	test RDI, RDI ; n == 0
	jz .zero

	mov [RBP - 8], RDI ; l_n = n
	dec RDI
	call faktorial ; ret faktorial(n-1)

	imul qword [RBP - 8] ; n * faktorial(n-1)


.zero:

	leave
	ret



	global is_digit
is_digit:
	enter 0,0

	mov RCX, 1
	mov RDX, 0
	mov RAX, 0

	cmp DIL, '0'
	cmovae RDX, RCX
	cmp DIL, '9'
	cmovbe RAX, RDX

	movzx RCX, SIL
	mov RDX, 0
	mov R8, 0

	cmp DIL, 'A'
	cmovae RDX, RCX
	cmp DIL, 'F'
	cmovbe R8, RDX

	or RAX, R8

	leave
	ret

; MINMAX CMOVcc
	global minmax
minmax:
	enter 0,0

	movsx RSI, ESI
	mov R10D, [RDI]
	mov R11D, [RDI]
	mov RAX, 0

.for:
	cmp RAX, RSI
	jge .endfor

	cmp R10D, [RDI + RAX * 4]
	cmovg R10D, [RDI + RAX * 4]

	cmp R11D, [RDI + RAX * 4]
	cmovl R11D, [RDI + RAX * 4]

	inc RAX
	jmp .for

.endfor
	mov [RDX], R10D
	mov [RCX], R11D
	leave 
	ret


; NASOBKY ---------------------------------
	global nasobky
nasobky:
	enter 0,0

	mov R9D, 0 ; citac = 0
	mov ECX, EDX ; cislo
	movsx RSI, ESI ; N
	mov R8, 0 ; i

.for:
	cmp R8, RSI
	jge .endfor

	mov EAX, [RDI + R8 * 4] ; pole[i]
	cdq
	idiv ECX
	cmp EDX, 0
	je .good
	jmp .back

.good:
	inc R9D

.back:
	inc R8
	jmp .for
.endfor:
	mov EAX, R9D
	leave
	ret

; IDIV -------------------------------
	global div_2int
div_2int:
	enter 0,0

	mov RCX, RDX
	mov EAX, EDI
	cdq
	idiv ESI 
	mov [RCX], EDX ;zbytek

	leave
	ret


; SKALAR ------------------------------
	global skalarni
skalarni:
	enter 0,0

	movsx RCX, EDX ; N 
	mov R11, 0 ; sum
	mov R10, 0 ; i
.for:
	cmp R10, RCX
	jge .endfor

	mov EAX, [RDI + R10 * 4]
	imul dword[RSI+ R10 * 4]
	add R11d, EAX

	inc R10
	jmp .for
.endfor:
	mov EAX, R11d
	leave
	ret

; MUL ------------------------------
	global mul_2char
mul_2char:
	movsx EDI, DIL
	movsx ESI, SIL

	global mul_2int
mul_2int:
	;enter 0,0

	mov EAX, 0
	mov EAX, EDI
	imul ESI

	shl RDX, 32
	or RAX, RDX

	;leave
	ret
