open util/ordering[State]

abstract sig Entity{
    eats: set Entity
}

one sig Farmer,Sheep,Wolf,Col extends Entity {
}

fact eating {
    eats = (Wolf->Sheep + Sheep->Col)
}

sig State {
    near: set Entity,
    far: set Entity
}

fact initialState {
    let s0 = first[] |
        s0.near = Entity && no s0.far
}

pred cross [from_i, from_o, to_i, to_o: set Entity]{
    (
        from_o = from_i - Farmer &&
        to_o = to_i - to_i.eats + Farmer
    )
    ||
    (   some item: from_i - Farmer|
            from_o = from_i - Farmer - item &&
            to_o = to_o - to_i.eats + Farmer + item
    )
}

pred cross2[from_i, from_o, to_i, to_o: set Entity] {
    some carried: lone (from_i - Farmer) |
        from_o = from_i - Farmer - carried and
        to_o = to_i - to_i.eats + Farmer + carried
}

fact stateTransition {
    all s1: State , s2: next[s1] |
        (Farmer in s1.near => 
            cross[s1.near, s2.near, s1.far, s2.far]
        )
        &&
        (Farmer in s1.far => 
            cross[s1.far, s2.far, s1.near, s2.near]
        )
}

pred solvePuzzle[] {
    last[].far = Entity
}
run solvePuzzle for 8 State 

