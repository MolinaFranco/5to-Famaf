-- {-# LANGUAGE LambdaCase #-}

module IntExp
  ( IntExp (..),
    Operators (..),
    interpIntExp,
  )
where
import State

{-
Gramaticas IntExp:
⟨intexp⟩ ::= ⟨natconst⟩
\| ⟨var⟩
\| - ⟨intexp⟩
\| ⟨intexp⟩ ⊕ ⟨intexp⟩
⊕ ∈ {+,−,∗, /, %,rem}
-}

data IntExp
  = Const Int
  | Var String
  | Neg IntExp
  | Operation Operators IntExp IntExp
  deriving (Eq, Show)

data Operators = Add | Sub | Mul | Div | Mod | Rem
  deriving (Eq, Show)

evalOp :: Operators -> Int -> Int -> Int
evalOp op = case op of
  Add -> (+)
  Sub -> (-)
  Mul -> (*)
  Div -> div
  Mod -> mod
  Rem -> rem


interpIntExp :: State -> IntExp -> Int
interpIntExp state expr = case expr of
  Const n -> n
  Var s -> state s
  Neg e -> -(interpIntExp state e)
  Operation op e1 e2 ->
    let v1 = interpIntExp state e1
        v2 = interpIntExp state e2
     in evalOp op v1 v2
