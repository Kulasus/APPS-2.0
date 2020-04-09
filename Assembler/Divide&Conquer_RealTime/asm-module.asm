	bits 64

	section .data

	section .text 

	global objem_valce
objem_valce:
	enter 0,0

	mov RAX, 3141
	mov R8D, 1000
	mov R9D, EDX
	imul EDI
	imul EDI
	imul R9D
	; v EAX mam ted ulozeny objem
	cdq
	idiv R8D
	; na prvni index v poli jsem ted ulozil vysledek po deleni
	mov [RSI + 0 * 4], EAX
	; na DRUHY index v poli jsem ted ulozil ZBYTEK
	mov [RSI + 1 * 4], EDX

	leave
	ret

	global hledej_rozptyl
hledej_rozptyl:
;  long l_odchylka = t_pole[ i ] - l_prumer_pole;
;  long l_rozptyl = l_odchylka * l_odchylka. 
;  RDI, RSI, RDX, RCX, R8 a R9
	enter 0,0
	mov RAX, 0
	mov R8, 0
	mov R9, 0 ;celkova suma 
;hledani prumeru
.forAvg:
	cmp R8, RSI
	jge .endforAvg	
	add R9, [RDI + R8 * 8]
	inc R8
	jmp .forAvg
.endforAvg:
	mov RAX, R9
	cdq
	idiv RSI
	; v RAX mam ted prumer
;-----------------------------------------
	mov R8, 0
	mov RCX, RAX ; presunu prumer do rcx at si muzu rax prepsat
	mov R9, [RDI] ; do R9 si dam prvni prvek v poli
	mov R10, 0 ;nejvetsi rozptyl
.forRozptyl:
	cmp R8, RSI
	jge .endforRozptyl
	mov RDX, [RDI + R8 * 8]
	sub RDX, RCX ; odchylku mam ted ulozenou v RDX
	mov RAX, RDX
	imul RDX ; ted mam rozptyl v RAX
	cmp RAX, R10
	jge .good
	inc R8
	jmp .forRozptyl
	
.good:
	mov R10, RAX
	mov R9, [RDI + R8 * 8]
	inc R8
	jmp .forRozptyl

.endforRozptyl:
	mov RAX, R9
	leave
	ret