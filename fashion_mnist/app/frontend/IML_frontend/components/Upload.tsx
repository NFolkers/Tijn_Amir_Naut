import axios from "axios";

// inspiration from: https://www.google.com/search?q=upload+file+button+in+react&oq=upload+file+button+in+react&gs_lcrp=EgZjaHJvbWUqBwgAEAAYgAQyBwgAEAAYgAQyCAgBEAAYFhgeMggIAhAAGBYYHtIBCDk5NzFqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8

interface UploadFileProps {
  updater: (step: string) => void;
  responseSet: (res: string) => void;
  setSelectedFile: (file: File | null) => void;
  selectedFile: File | null;
}

const UploadFile = ({
  updater,
  responseSet,
  setSelectedFile,
  selectedFile,
}: UploadFileProps) => {
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUpload = () => {
    if (!selectedFile) {
      console.log("Please upload a file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);
    axios
      .post("http://127.0.0.1:8000/upload", formData)
      .then((response) => {
        responseSet(response.data);
      })
      .catch((error) => {
        console.log(error);
      });
    updater("response");
  };

  return (
    <div>
      <h3>Upload File</h3>
      <input type="file" onChange={handleFileUpload} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default UploadFile;
