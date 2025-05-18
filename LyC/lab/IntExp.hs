
-- {-# LANGUAGE LambdaCase #-}

module IntExp
  ( 
    IntExp
  )
where

{-
Gramaticas IntExp:
⟨intexp⟩ ::= ⟨natconst⟩
| ⟨var⟩
| - ⟨intexp⟩
| ⟨intexp⟩ ⊕ ⟨intexp⟩
⊕ ∈ {+,−,∗, /, %,rem}
-}

type Var = String

data IntExp 
  = Int
  | Var
  | Neg Int
  | BinOperator IntExp IntExp
  deriving (Eq, Show)

data BinOperator = Add | Sub | Mul | Div | Mod | Rem
    deriving (Eq, Show)

type State = Var -> Integer

interpIntExp :: State -> IntExp -> Integer
interpIntExp state expr = case expr of
  Const n           -> n
  Var x             -> env x
  Neg e             -> negate (evalIntExp env e)
  BinOp op e1 e2    ->
    let v1 = evalIntExp env e1
        v2 = evalIntExp env e2
    in evalOp op v1 v2


