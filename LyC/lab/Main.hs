module Main where

import BoolExp
import Comm
import Data.Map qualified as M
import IntExp
import State
import System.IO (hFlush, stdout)
import Text.Read (readMaybe)

main :: IO ()
main = loop emptyState

loop :: State -> IO ()
loop st = do
  putStr "> "
  hFlush stdout
  input <- getLine
  if input == "exit"
    then putStrLn "Bye!"
    else case parseComm input of
      Just comm -> do
        let newState = interpComm st comm
        print (dumpState newState)
        loop newState
      Nothing -> do
        putStrLn "Invalid command."
        loop st

-- Estado vacío (todo variable da 0)
emptyState :: State
emptyState = const 0

-- Para mostrar el estado (solo si usás Map)
dumpState :: State -> M.Map String Int
dumpState st = M.fromList [(v, st v) | v <- ["x", "y", "z"]] -- personalizá esto

-- Parser muy simple
parseComm :: String -> Maybe Comm
parseComm str = case words str of
  [var, ":=", valStr] ->
    case readMaybe valStr of
      Just n -> Just (Assign var (Const n))
      Nothing -> Nothing
  ["skip"] -> Just Skip
  _ -> Nothing
