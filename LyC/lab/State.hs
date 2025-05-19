module State
  ( State,
    updateState,
  )
where

type State = String -> Int

updateState :: String -> Int -> State -> State
updateState name val oldState query =
  if query == name
    then val
    else oldState query
