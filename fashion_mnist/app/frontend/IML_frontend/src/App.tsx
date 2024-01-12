import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import Upload from "../components/Upload";

function App() {
  const [count, setCount] = useState(0);

  return (
    <>
      <h1>Welcome to our Fashion Mnist model API</h1>
      <div className="card">
        <Upload></Upload>
      </div>
    </>
  );
}

export default App;
