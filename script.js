document.addEventListener("DOMContentLoaded", () => {
  loadRoom("./rooms/lotus-design-n-print-8qNuR1lIv_k-unsplash.jpg", 11);

  // Room upload button logic
  document.getElementById("room-upload-btn").addEventListener("click", () => {
      document.getElementById("room-upload").click();
  });
  document.getElementById("room-upload").addEventListener("change", handleRoomUpload);

  // Texture upload button logic
  document.querySelector(".upload-texture").addEventListener("click", () => {
      document.getElementById("texture-upload").click();
  });
  document.getElementById("texture-upload").addEventListener("change", handleTextureUpload);
});

function loadRoom(roomImage, textureCount) {
  const roomImgElement = document.getElementById("room-image");
  roomImgElement.src = roomImage;

  const textureContainer = document.getElementById("textures");
  textureContainer.innerHTML = ""; // Clear previous textures

  for (let i = 1; i <= textureCount; i++) {
      const textureDiv = document.createElement("div");
      textureDiv.className = "texture";
      textureDiv.style.backgroundImage = `url('./textures/texture${i}.jpeg')`;
      textureContainer.appendChild(textureDiv);
  }

  // Re-add the "+" upload tile
  const uploadTile = document.createElement("div");
  uploadTile.className = "texture upload-texture";
  uploadTile.innerText = "+";
  uploadTile.addEventListener("click", () => {
      document.getElementById("texture-upload").click();
  });
  textureContainer.appendChild(uploadTile);
}

function handleRoomUpload(event) {
  const file = event.target.files[0];
  if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
          document.getElementById("room-image").src = e.target.result;
      };
      reader.readAsDataURL(file);
  }
}

function handleTextureUpload(event) {
  const file = event.target.files[0];
  if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
          const textureDiv = document.createElement("div");
          textureDiv.className = "texture";
          textureDiv.style.backgroundImage = `url('${e.target.result}')`;

          const uploadTile = document.querySelector(".upload-texture");
          document.getElementById("textures").insertBefore(textureDiv, uploadTile);
      };
      reader.readAsDataURL(file);
  }
}
