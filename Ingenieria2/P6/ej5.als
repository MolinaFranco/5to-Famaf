// Ejercicio 5. Modele relaciones binarias en Alloy. Utilizando predicados, especifique que una relaci´on
// es:

sig Elem {}

sig Relacion {
    r: Elem -> Elem
}

fun idenElem[]: (Elem -> Elem){
	(iden & (Elem -> Elem))
}

fun univRel[]: (Elem -> Elem){
	(Elem -> Elem)
}

pred reflexiva[R: Relacion]{
	idenElem in R.r
}

pred transitivaMia[R: Relacion]{
    all e1, e2, e3: Elem | 
        (e1 -> e2 in R.r and e2 -> e3 in R.r) => (e1 -> e3 in R.r)
}

pred transitiva[R: Relacion]{
	(R.r).(R.r) in R.r
}

pred sim[R: Relacion]{
	(R.r in ~(R.r))
}

pred asim[R:Relacion]{
	no (R.r & ~(R.r))
}

pred totalidad[R: Relacion]{
	(R.r + ~(R.r)) = univRel
}

// (a) un preorden: reflexivo y transitivo

pred isPreorder(R: Relacion){
    all e: Elem | e -> e in R.r
    all e1, e2, e3: Elem | 
        (e1 -> e2 in R.r and e2 -> e3 in R.r) => (e1 -> e3 in R.r)
}

pred preorden[r: Relacion]{
	reflexiva[r] 
	transitiva[r]
}

// (b) un orden parcial, refelx y trans y ademas asim

pred poset[r: Relacion]{
	preorden[r] 
	asim[r]
}

// (c) un orden total,

pred totalorden[r: Relacion]{
    poset[r]
    totalidad[r]
}

// (d) un orden estricto,

pred strictorden[r:Relacion]{
    not reflexiva[r]
    asim[r]
    transitiva[r]
}

// (e) que tiene primer elemento,

pred primerElemento[R:Relacion]{
    one p: Elem | all e: Elem - p |
        e -> p not in R.r
        // none (e->p & R.r)
}

pred primerElemento2[R:Relacion]{
    one p: Elem | 
        all y: Elem | 
            p!=y implies (p -> y in R.r and not y -> p in R.r)

} 

// (f) que tiene ´ultimo elemento.

pred ultimoElemento[R:Relacion]{
    one p: Elem | 
        all y: Elem | 
            p!=y implies (y -> p in R.r and not y -> p in R.r)

} 

//////////////////////////////////////////////////////
// Escriba aserciones para las siguientes propiedades:
//////////////////////////////////////////////////////


// todo orden parcial es total;

assert parcialEsTotal{
    all r: Relacion | poset[r] implies totalorden[r]
}

// todo orden parcial tiene primer elemento;

assert parcialTienePrimer{
    all r: Relacion | poset[r] implies primerElemento[r]
}

// todo orden total con primer elemento x y ´ultimo elemento y satisface x != y;

// assert primerultimo{
    // all r: Relacion | totalorden[r] and primer
// }

// la uni´on de ´ordenes estrictos es un orden estricto;

assert unionEstrictos{
	all r, s: Relacion | strictorden[r] and strictorden[s] => strictorden[r+s]
}

// la composici´on de ´ordenes estrictos es un orden estricto.

assert composicionEstrictos {
    all a, b: Relacion | 
        strictorden[a] and strictorden[b] => {
            let c = Relacion | c.r = (a.r) . (b.r) and strictorden[c]
        }
}

check composicionEstrictos for 5 but exactly 2 Relacion
check parcialTienePrimer for 5 but exactly 1 Relacion