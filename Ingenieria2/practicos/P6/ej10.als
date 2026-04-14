// Ejercicio 10. Modele el problema de las torres de Hanoi en Alloy y use el Alloy Analyzer para
// encontrar una soluci´on

open util/ordering[State]


abstract sig Disk {
    gt: set Disk 
}

one sig D1, D2, D3, D4 extends Disk {}

abstract sig Tower {}

one sig T1, T2, T3 extends Tower {}

fact order {
    gt = (
        D4->D3+D4->D2+D4->D1+
        D3->D2+D3->D1+
        D2->D1
    )
}

sig State {
    content: Tower -> set Disk
} {
    all d: Disk | one t: Tower | d in content[t]
}

fact initialState {
    let s0 = first[] | (
        s0.content[T1] = Disk and
        no s0.content[T2] and
        no s0.content[T3] 
        )
}


pred move[s1, s2: State] {
    one d: Disk, from, to: Tower |
        d in s1.content[from]
        and s2.content[from] = s1.content[from] - d
        and s2.content[to] = s1.content[to] + d
        and all t: Tower - from - to |
            s1.content[t] = s2.content[t]

        and all d2: s1.content[from] - d | d in d2.gt
        and all d2: s2.content[to] - d | d in d2.gt
}




fact stateTransition {
    all s1: State , s2: next[s1] |
        move[s1,s2]
        // s1.content != s2.content implies move[s1, s2]
}

pred solvePuzzle[] {
    last[].content[T3] = Disk
}
run solvePuzzle for 10 State 
