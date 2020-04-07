bits 64

	section .data

    section .text

	global kopiruj_velka
kopiruj_velka:
	enter 0,0

	push RBX
	mov RAX, 0
	mov RCX, 0

.for:
	cmp byte [RSI + RCX], 0
	je .endfor
	cmp byte [RSI + RCX], 64
	jbe .bad
	cmp byte [RSI + RCX], 91
	jae .bad
	mov BL, [RSI + RCX]
	mov [RDI + RCX], BL
	inc RAX
	jmp .back

.bad:
	mov [RDI + RCX], DL
	jmp .back

.back:
	inc RCX
	jmp .for

.endfor:
	pop RBX
	leave
	ret
;------------------------------------------------
	global bitova_maska
bitova_maska:
	enter 0,0

	push RBX
	mov RAX, 0
	mov RCX, 0
	mov RDX, 0

.for:
	cmp RDX, RSI
	je .endfor
	mov RCX, 1
	mov BL, byte [RDI + RDX]
.for2:
	cmp byte BL, 0
	je .back
	shl RCX, 1
	dec BL
	jmp .for2

.back:
	add RAX, RCX
	inc RDX
	jmp .for

.endfor:
	pop RBX
	leave
	ret

;------------------------------------------------
	global over_format
over_format:
	enter 0,0

	mov RAX, 1
	mov RDX, 0
	mov RCX, 0
.for:
	cmp byte [RDI + RCX], 0
	je .endfor
	cmp byte [RDI + RCX], 45
	je .minus
	cmp byte [RDI + RCX], 46
	je .tecka
	cmp byte [RDI + RCX], 95
	je .podtrzitko
	cmp byte [RDI + RCX], 48
	jb .bad
	cmp byte [RDI + RCX], 57
	ja .bad
	jmp .good	

.minus:
	cmp RCX, 0
	je .good
	jmp .bad

.tecka:
	cmp RDX, 0
	jne .bad
	cmp byte [RDI + RCX - 1],48
	jb .bad
	cmp byte [RDI + RCX - 1],57
	ja .bad
	cmp byte [RDI + RCX + 1],48
	jb .bad
	cmp byte [RDI + RCX + 1],57
	ja .bad
	inc RDX
	jmp .good

.podtrzitko:
	cmp byte [RDI + RCX - 1],48
	jb .bad
	cmp byte [RDI + RCX - 1],57
	ja .bad
	cmp byte [RDI + RCX + 1],48
	jb .bad
	cmp byte [RDI + RCX + 1],57
	ja .bad
	jmp .good

.good:
	inc RCX
	jmp .for

.bad:
	mov RAX, 0

.endfor:
	leave
	ret

;------------------------------------------------
; RDI, RSI, RDX, RCX, R8 a R9
	global najdi_pozice_minmax
najdi_pozice_minmax:
	enter 0,0
	push RBX
	mov RCX, 0
	mov RAX, [RDI]
	mov RBX, [RDI]
	mov R8, 0
	mov R9, 0
.for:
	cmp RCX, RSI
	je .endfor
	cmp [RDI + RCX * 8], RAX
	jg .greater
	cmp [RDI + RCX * 8], RBX
	jl .lower
	jmp .continue

.greater:
	mov RAX, [RDI + RCX * 8]
	mov R8, RCX
	jmp .continue

.lower:
	mov RBX, [RDI + RCX * 8]
	mov R9, RCX
	jmp .continue

.continue:
	inc RCX
	jmp .for

.endfor:
	mov [RDX], R8
	mov [RDX + 8], R9
	pop RBX
	leave
	ret