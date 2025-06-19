document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("fileUpload");
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const cameraImage = document.getElementById("cameraImage");

  // Aktifkan kamera belakang
  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => {
      console.warn("Kamera tidak tersedia:", err.message);
    });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      cameraImage.value = "";
    }
  });
});

function captureImage() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const cameraImage = document.getElementById("cameraImage");

  const minSide = Math.min(video.videoWidth, video.videoHeight);
  const startX = (video.videoWidth - minSide) / 2;
  const startY = (video.videoHeight - minSide) / 2;

  canvas.width = 500;
  canvas.height = 500;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, startX, startY, minSide, minSide, 0, 0, 500, 500);

  const dataURL = canvas.toDataURL("image/jpeg");
  cameraImage.value = dataURL;

  alert("📸 Gambar berhasil diambil! Klik 'Klasifikasi' untuk melanjutkan.");
}
