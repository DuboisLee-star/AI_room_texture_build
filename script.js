document.addEventListener("DOMContentLoaded", () => {
  const textureContainer = document.getElementById("textures");
  const textureCount = 11;

  for (let i = 1; i <= textureCount; i++) {
      const textureDiv = document.createElement("div");
      textureDiv.className = "texture";
      textureDiv.style.backgroundImage = `url('./textures/texture${i}.jpeg')`;
      textureContainer.appendChild(textureDiv);
  }
});
