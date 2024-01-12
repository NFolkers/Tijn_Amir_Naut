import { useState } from "react";
import "./App.css";
import UploadFile from "../components/Upload";

function App() {
  const [step, setStep] = useState("home");
  const [response, setResponse] = useState("");
  const [output, setOuput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleStepChange = (str: string) => {
    setStep(str);
  };
  const handleResponseChange = (res: string) => {
    setResponse(res);
    console.log(res);
    // console.log(JSON.stringify(res.prediction));
    if (res.prediction == "null") {
      handleStepChange("error");
      setOuput(JSON.stringify(res.error));
    } else {
      setOuput(JSON.stringify(res.prediction));
    }
  };

  const handleFlush = () => {
    setOuput("");
    setSelectedFile(null);
    setResponse("");
    setStep("home");
  };
  // useEffect(() => {

  // }, [output, step]);

  let content;
  if (step == "home") {
    content = (
      <>
        <UploadFile
          updater={handleStepChange}
          responseSet={handleResponseChange}
          setSelectedFile={setSelectedFile}
          selectedFile={selectedFile}
        ></UploadFile>
      </>
    );
  }

  if (step == "response") {
    content = (
      <>
        <h2>Our model predict that you image depicts a: </h2>
        <h1>{output}</h1>
        <button onClick={handleFlush}>Try a different image</button>
      </>
    );
  }
  if (step == "error") {
    content = (
      <>
        <h3>{output}</h3>
        <button onClick={handleFlush}>Try again</button>
      </>
    );
  }

  return (
    <>
      <h1>Welcome to our Fashion Mnist model API</h1>
      <div className="card">{content}</div>
    </>
  );
}

export default App;
