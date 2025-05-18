
-- {-# LANGUAGE LambdaCase #-}

module Comm
  ( 
    Comm
  )
where

{-
Gramaticas Comandos:
⟨comm⟩ ::= skip
| ⟨var⟩ := ⟨intexp⟩
| ⟨comm⟩ ; ⟨comm⟩
| if ⟨boolexp⟩ then ⟨comm⟩ else ⟨comm⟩
| newvar ⟨var⟩ := ⟨intexp⟩ in ⟨comm⟩
| while ⟨boolexp⟩ do ⟨comm⟩
| fail
-}

type Var   = String

data Comm 
  = Skip
  | Assign Var IntExp
  | Seq Comm Comm
  | Condicional Bool Comm Comm
  | NewVar Var Int Comm
  | While Bool Comm
  | Fail
  deriving (Eq, Show)

