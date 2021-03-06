; 	This defines in which mode will nasm compile the code
	bits 64
; 	Data section is used for declaring variables
	section .data

; Variables
; 	extern means variables are not declared in this file
;-----------------------------------------------------------------
;   VARIABLES FOR PART 1
;-----------------------------------------------------------------
	extern g_char, g_num32, g_num64
	extern g_array_char, g_array_num32, g_array_num64
	extern g_index, g_index_plus_one
	extern g_firstIndexOfArray, g_lastIndexOfArray
	extern g_switchIndex1, g_switchIndex2

;-----------------------------------------------------------------
;   VARIABLES FOR PART 2
;-----------------------------------------------------------------
	extern g_rgb565_b1, g_rgb565_b2
	extern g_char_array, g_char_mean
	extern g_number
	extern g_int_array, g_int_array_sum
	extern g_positive, g_negative
	extern g_string_len, g_string
	extern g_minimum, g_maximum
	extern g_minimum_pos, g_maximum_neg
	extern g_string_one, g_counter, g_number_to_find
	extern g_brackets, g_brackets_left, g_brackets_right
	extern g_banan, g_jabko
	extern g_bin_num, g_bin_num_char
; Text section is used for defining functions
	section .text
;*****************************************************************
;-----------------------------------------------------------------
;   FUNCTIONS FOR PART 1
;-----------------------------------------------------------------
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
	; Definition of function switch_between_array_num32_array_num64_in_asm
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
	; Definition of function switch_first_and_last_index_array_num32_in_asm
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
	; Definition of function switch_neighbors_array_num32_in_asm
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
	; Definition of function set_array_num32_neighbors_value
	global set_array_num32_neighbors_value
set_array_num32_neighbors_value:
	enter 0,0

	mov EAX, [g_index]
	mov ECX, [g_index_plus_one]
	mov EBX, [g_array_num32 + ECX * 4]
	mov [g_array_num32 + EAX * 4], EBX

	leave
	ret

;******************************************************************
	; Definition of function switch_two_values_index_num32_in_asm	
	global switch_two_values_index_num32_in_asm
switch_two_values_index_num32_in_asm:
	enter 0,0

	mov EAX, [g_switchIndex1]
	mov EBX, [g_switchIndex2]
	mov ECX, [g_array_num32 + EAX * 4]
	mov EDX, [g_array_num32 + EBX * 4]
	mov [g_array_num32 + EAX * 4], EDX
	mov [g_array_num32 + EBX * 4], ECX

	leave
	ret

;-----------------------------------------------------------------
;   FUNCTIONS FOR PART 2
;-----------------------------------------------------------------
;******************************************************************
	; Definition of function move_blue
	global move_blue
move_blue:
	enter 0,0

	mov AX, 0b11111         ; blue mask -> 0b11111 = 0b0000000000011111
	mov CX, [g_rgb565_b1] 
	and CX, AX				; only blue
	not AX					; mask for green and red
	and [g_rgb565_b2], AX	; only red and green
	or [g_rgb565_b2], CX

	leave
	ret
;******************************************************************
	; Definition of function mean_char_array
	global mean_char_array
mean_char_array:
	enter 0,0

	movsx RAX, byte [g_char_array]  ; sum
	movsx RCX, byte [g_char_array + 1]
	add RAX, RCX 					; sum += g_char_array[1]
	movsx RCX, byte [g_char_array + 2]
	add RAX, RCX 					; sum += g_char_array[2]
	movsx RCX, byte [g_char_array + 3]
	add RAX, RCX 					; sum += g_char_array[3]

	shr RAX, 2						; sum / 4
	mov [g_char_mean], RAX

	leave
	ret
;******************************************************************
	; Definition of function number_mul100
	global number_mul100
number_mul100:
	enter 0,0

	mov ECX, [g_number]
	shl ECX, 2				; *4
	mov EAX, ECX
	shl EAX, 3				; *4*8
	add ECX, EAX
	shl EAX, 1				; *4*8*2
	add ECX, EAX
	mov [g_number], ECX 

	leave
	ret

;******************************************************************
	; Definition of function sum_int_array
	global sum_int_array
sum_int_array:
	enter 0,0
	
	xor EAX, EAX   		; mov EAX, 0 / sub EAX, EAX
	; for (rdx = 0; rdx < 10; rdx++){}
	mov RDX, 0
.back:
	cmp RDX, 10 		; sub rdx, 10
	jge .endfor

	add EAX, [g_int_array + RDX * 4]

	inc RDX
	jmp .back
	; endfor
.endfor:
	mov [g_int_array_sum], EAX
	leave
	ret


;******************************************************************
	; Definition of function posneg_int_array
	global posneg_int_array
posneg_int_array:
	enter 0,0

	mov RDX, 0
.back:
	cmp RDX, 10 		
	jge .endfor

	cmp dword [g_int_array + RDX * 4], 0
	je .continue
	jg .positive
	inc dword [g_negative]
	jmp .continue

.positive:
	inc dword [g_positive]
	jmp .continue
.continue:
	inc RDX
	jmp .back
.endfor:
	leave
	ret


;******************************************************************
	; Definition of function string_len
	global string_len
string_len:
	enter 0,0

	mov RCX, g_string ; -> no []? it means pointers bruh ;)

.back:
	cmp byte [RCX], 0
	je .end
	inc RCX
	jmp .back

.end:
	sub RCX, g_string
	mov [g_string_len], ECX

	leave
	ret

;******************************************************************
	; Definition of function string_low
	global string_low
string_low:
	enter 0,0

	mov RCX, 0

.back:
	cmp byte [g_string + RCX], 0
	je .end
	cmp byte [g_string + RCX], 'A'
	jb .continue
	cmp byte [g_string + RCX], 'Z'
	ja .continue
	add byte [g_string + RCX], 'a' - 'A'


.continue:
	inc RCX
	jmp .back

.end:
	sub RCX, g_string
	mov [g_string_len], ECX


	leave
	ret

;******************************************************************
	; Definition of function find_minimum
	global find_minimum
find_minimum:
	enter 0,0

	mov EAX, 0
	mov EDX, 0

.back:
	cmp EDX, 9
	jge .endfor
	mov EDI, EDX
	add EDI, 1
	mov ESI, [g_int_array + EDI * 4]
	cmp EAX, ESI
	jl .continue

.isMinimum:
	mov EAX, ESI

.continue:
	inc EDX
	jmp .back

.endfor:
	mov [g_minimum], EAX
	leave
	ret


;******************************************************************
	; Definition of function find_maximum
	global find_maximum
find_maximum:
	enter 0,0

	
	mov EAX, 0
	mov EDX, 0

.back:
	cmp EDX, 9
	jge .endfor
	mov EDI, EDX
	add EDI, 1
	mov ESI, [g_int_array + EDI * 4]
	cmp EAX, ESI
	jg .continue

.isMaximum:
	mov EAX, ESI

.continue:
	inc EDX
	jmp .back

.endfor:
	mov [g_maximum], EAX
	leave
	ret

;******************************************************************
	; Definition of function find_maximum_neg
	global find_maximum_neg
find_maximum_neg:
	enter 0,0

	call find_minimum
	mov EDX, 0 ; i

.back:
	cmp EDX, 9
	jge .endfor
	mov EDI, EDX
	add EDI, 1
	mov ESI, [g_int_array + EDI * 4] 
	cmp ESI, 0
	jge .continue
	cmp EAX, ESI ; EAX -> max | ESI ->i+1
	jg .continue 

.isMaximumNeg:
	mov EAX, ESI

.continue:
	inc EDX
	jmp .back

.endfor:
	mov [g_maximum_neg], EAX
	leave
	ret

;******************************************************************
; Definition of function find_minimum_pos
	global find_minimum_pos
find_minimum_pos:
	enter 0,0

	call find_maximum
	mov EDX, 0 ; i

.back:
	cmp EDX, 9
	jge .endfor
	mov EDI, EDX
	add EDI, 1
	mov ESI, [g_int_array + EDI * 4] 
	cmp ESI, 0
	jle .continue
	cmp EAX, ESI ; EAX -> min | ESI ->i+1
	jl .continue 

.isMinimumPos:
	mov EAX, ESI

.continue:
	inc EDX
	jmp .back

.endfor:
	mov [g_minimum_pos], EAX
	leave
	ret

;******************************************************************
; 			Definition of function count_of_number
	global count_of_number
count_of_number:
	enter 0,0

	mov RDX, 0
	mov ECX, 0
	mov AL, [g_number_to_find]

.back:
	cmp byte [g_string_one + RDX], 0
	je .end
	cmp byte [g_string_one + RDX], AL
	je .counter
	
.continue:
	inc RDX
	jmp .back

.counter:
	inc ECX
	inc RDX
	jmp .back

.end:
	mov [g_counter], ECX
	leave
	ret

;******************************************************************
; 			Definition of function check_brackets
	global check_brackets
check_brackets:
	enter 0,0

	mov RDX, 0
	mov EAX, 0 
	mov EBX, 0

.back:
	cmp byte [g_brackets + RDX], 0
	je .end
	cmp byte [g_brackets + RDX], '('
	je .counter1
	cmp byte [g_brackets + RDX], ')'
	je .counter2


.continue:
	inc RDX
	jmp .back

.counter2:
	inc EBX
	inc RDX
	jmp .back

.counter1:
	inc EAX
	inc RDX
	jmp .back

.end:
	mov [g_brackets_left], EAX
	mov [g_brackets_right], EBX
	leave
	ret

;******************************************************************
; 			Definition of function change_string
	global change_string
change_string:
	enter 0,0

	mov RCX, 0
	mov AL, [g_banan]
	mov DL, [g_jabko]

.back:
	cmp byte [g_string_one + RCX], 0
	je .end
	cmp byte [g_string_one + RCX], AL
	je .change

.continue:
	inc RCX
	jmp .back

.change:
	mov byte [g_string_one + RCX], DL
	inc RCX
	jmp .back

.end:
	mov [g_counter],RDX
	leave
	ret


;******************************************************************
; 			Definition of function num_to_bin_char
	global num_to_bin_char
num_to_bin_char:
	enter 0,0

	mov RAX, 7
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

;******************************************************************
; 			Definition of function num_to_hexa_char
	global num_to_hexa_char
num_to_hexa_char:
	enter 0,0

	mov RAX, 7
	mov RCX, [g_bin_num]
	

.back:
	cmp RCX, 0
	je .end
	mov RDX, RCX
	and RDX, 15	; here is reminder
	shr RCX, 4	; shifting te number
	cmp RDX, 10
	je .A
	cmp RDX, 11
	je .B
	cmp RDX, 12
	je .C
	cmp RDX, 13
	je .D
	cmp RDX, 14
	je .E
	cmp RDX, 15
	je .F
	add RDX, 48
	mov [g_bin_num_char + RAX], DL
	jmp .continue

.A:
mov byte [g_bin_num_char + RAX], 'A'
.B:
mov byte [g_bin_num_char + RAX], 'B'
.C:
mov byte [g_bin_num_char + RAX], 'C'
.D:
mov byte [g_bin_num_char + RAX], 'D'
.E:
mov byte [g_bin_num_char + RAX], 'E'
.F:
mov byte [g_bin_num_char + RAX], 'F'
	

.continue:
	dec RAX
	jmp .back

.end:
	leave
	ret
