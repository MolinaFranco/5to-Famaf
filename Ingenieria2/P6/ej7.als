// Ejercicio 7. Se desea modelar en Alloy la manipulaci´on de cat´alogos de m´usica. 
// Estos cat´alogos contienen canciones, interpretes y el listado de canciones interpretadas. 
// Esta estructura podr´ıa modelarse
// de la siguiente manera:

// Se dice que un cat´alogo es consistente si todas las canciones del cat´alogo est´an registradas por
// alg´un interprete y todo int´erprete del cat´alogo tiene registrada alguna canci´on. Complete los tres
// puntitos de la definici´on de Catalogo para que asegure consistencia.

sig Interprete {}

sig Cancion {}

sig Catalogo {
    canciones: set Cancion,
    interpretes: set Interprete,
    interpretaciones: canciones -> interpretes
}{
    canciones = ~interpretaciones[interpretes]
    canciones = interpretaciones . interpretes

    // a -> b -> c y te queres quedar con a->c
    // x = (x.univ) . (univ.x)

    interpretes = interpretaciones[canciones] 
}

// Defina adem´as lo siguiente:
// (a) Un predicado que dado un cat´alogo y una cacni´on con su interprete, devuelva un nuevo cat´alogo
// igual al primero pero con esa interpretaci´on agragada.

pred getCancion[c_i, c_o: Catalogo, s: Cancion, i: Interprete]{
    c_o.interpretaciones = c_i.interpretaciones ++ (s -> i)
}

// (b) Un predicado que dado un cat´alogo y una canci´on con su interprete, devuelva un nuevo cat´alogo
// igual al primero pero eliminando esa interpretaci´on.

pred delete[ci, co: Catalogo, s: Cancion, i: Interprete] {
	co.interpretaciones = ci.interpretaciones - (s -> i)
}

// (c) Una funci´on que, dado un cat´alogo, devuelva los pares de interpretes que interpretan la misma
// canci´on.

fun coInterprete[c: Catalogo]: (Interprete -> Interprete) {	
	~(c.interpretaciones).(c.interpretaciones) 
	- (iden & (Interprete -> Interprete))
}

// con la primera parte i->c->i con el . i->i y despues elimino los que son iguales

run delete for 3 but 2 Catalogo

// Haga el mejor uso posible del c´alculo relacional.