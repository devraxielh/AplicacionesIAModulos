import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResponse("");
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    try {
      setLoading(true);
      const res = await axios.post("http://127.0.0.1:8000/img_txt", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResponse(res.data.response);
    } catch (err) {
      setResponse("❌ Error: " + err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 30 }}>
      <h2>Imagen a texto</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <br /><br />
      <button onClick={handleSubmit} disabled={!file || loading}>
        {loading ? "Procesando..." : "Enviar"}
      </button>
      <br /><br />
      {response && <p><strong> Descripción:</strong> {response}</p>}
    </div>
  );
}

export default App;