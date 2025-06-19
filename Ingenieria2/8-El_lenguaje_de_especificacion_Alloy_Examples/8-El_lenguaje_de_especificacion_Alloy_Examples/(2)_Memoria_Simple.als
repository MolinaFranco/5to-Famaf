
open util/ordering [ State ] 

// El dominio del problema ---------

sig Addr, Data { }

sig Memory{
    data: Addr -> lone Data
   }

pred write [ m_i, m_o: Memory, a: Addr, d: Data ] {
    m_o.data = m_i.data ++ a -> d
   }

pred read [ m: Memory, a: Addr, d_o: Data ] {
    let d = m.data[a] | some d implies d = d_o
}

// La dinámica --------------------
// Espacio de estado

sig State {
    mem:Memory
   }

// Estado inicial (predicado auxiliar)

pred init [ m: Memory ] { no m.data }

// Traza

fact Traces {
    init[first[].mem]
    all s1:State, s2:next[s1] |
      (some a:Addr, d:Data | read[s1.mem,a,d] and s1 = s2)
      or
      (some a:Addr, d:Data | write[s1.mem,s2.mem,a,d])
   }

// Propiedad requerida

pred freshDir [ m: Memory ] {
    some a: Addr | no (m.data.Data & a) 
   }

pred freshDirInLast {
    freshDir [last[].mem]
   }

run freshDirInLast for 5
