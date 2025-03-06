$(document).ready(() => {
    // Load the room and textures
    loadRoom("./rooms/lotus-design-n-print-8qNuR1lIv_k-unsplash.jpg", 11);

    // Room upload button logic
    $("#room-upload-btn").on("click", () => {
        $("#room-upload").click();
    });

    $("#room-upload").on("change", handleRoomUpload);

    // Texture upload button logic
    $(".upload-texture").on("click", () => {
        $("#texture-upload").click();
    });

    $("#texture-upload").on("change", handleTextureUpload);
});

function loadRoom(roomImage, textureCount) {
    $("#room-image").attr("src", roomImage);

    const textureContainer = $("#textures");
    textureContainer.empty(); // Clear previous textures

    for (let i = 1; i <= textureCount; i++) {
        const textureDiv = $("<div>")
            .addClass("texture")
            .css("background-image", `url('./textures/texture${i}.jpeg')`);
        textureContainer.append(textureDiv);
    }

    // Re-add the "+" upload tile
    const uploadTile = $("<div>").addClass("texture upload-texture").text("+");
    uploadTile.on("click", () => {
        $("#texture-upload").click();
    });
    textureContainer.append(uploadTile);
}

function handleRoomUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append("file", file);

        $.ajax({
            url: "http://localhost:8000/process-image/",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: (data) => {
                if (data.processed_image) {
                    $("#room-image").attr(
                        "src",
                        `data:image/png;base64,${data.processed_image}`
                    );
                }
            },
            error: (error) => {
                console.error("Error processing image:", error);
                alert(
                    "There was an issue connecting to the backend or processing the image."
                );
            },
        });
    }
}

function handleTextureUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const textureDiv = $("<div>")
                .addClass("texture")
                .css("background-image", `url('${e.target.result}')`);
            $(".upload-texture").before(textureDiv);
        };
        reader.readAsDataURL(file);
    }
}
