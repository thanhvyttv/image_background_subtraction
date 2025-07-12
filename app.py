import io  # Để đọc dữ liệu ảnh từ upload

import cv2
import numpy as np
import streamlit as st
from PIL import Image  # Dùng để xử lý ảnh từ Streamlit uploader

st.set_page_config(layout="wide")  # Tối ưu hóa không gian hiển thị


# --- Hàm GrabCut ---
def apply_grabcut_and_blend(image_np, bbox, new_background_np):
    # Tạo mask ban đầu
    mask = np.zeros(image_np.shape[:2], np.uint8)

    # Định nghĩa các mảng cho GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Chạy GrabCut
    cv2.grabCut(image_np, mask, bbox, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Tạo mask cuối cùng
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        "uint8"
    )

    # Tách vật thể với kênh Alpha
    object_bgr = cv2.bitwise_and(image_np, image_np, mask=mask2)
    object_rgba = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2BGRA)
    object_rgba[:, :, 3] = mask2  # Gán kênh alpha

    # new_background_np đã được resize để khớp với image_np ở bên ngoài hàm này
    # nên không cần resize lại ở đây.
    # new_background_resized = cv2.resize(
    #     new_background_np, (image_np.shape[1], image_np.shape[0])
    # )

    # Ghép vật thể vào background mới bằng alpha blending
    alpha_normalized = mask2.astype(float) / 255.0
    alpha_normalized_3_channels = cv2.merge(
        [alpha_normalized, alpha_normalized, alpha_normalized]
    )

    foreground = cv2.multiply(
        object_rgba[:, :, :3].astype(float), alpha_normalized_3_channels
    )
    background = cv2.multiply(
        new_background_np.astype(float),
        1.0 - alpha_normalized_3_channels,  # Sử dụng new_background_np đã xử lý
    )
    final_result = cv2.add(foreground, background).astype(np.uint8)

    return final_result, object_bgr, mask2


# --- Giao diện Streamlit ---
st.title("Ứng Dụng Ghép Ảnh (Sử dụng GrabCut)")
st.write(
    "Tách vật thể từ một ảnh và ghép vào ảnh nền mới. Bạn cần nhập tọa độ hình chữ nhật bao quanh vật thể."
)

# --- Upload ảnh có vật thể ---
st.sidebar.header("1. Tải lên ảnh có vật thể")
uploaded_object_file = st.sidebar.file_uploader(
    "Chọn ảnh có vật thể (JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    key="object_uploader",
)
raw_object_image_np = None  # Lưu ảnh gốc chưa resize
if uploaded_object_file is not None:
    bytes_data = uploaded_object_file.getvalue()
    object_image_pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    raw_object_image_np = np.array(object_image_pil)
    raw_object_image_np = cv2.cvtColor(raw_object_image_np, cv2.COLOR_RGB2BGR)
    st.image(
        object_image_pil,
        caption="Ảnh gốc có vật thể (chưa xử lý)",
        use_container_width=True,
    )
    st.write(f"Kích thước ảnh gốc ban đầu: {raw_object_image_np.shape}")

# --- Upload ảnh nền mới ---
st.sidebar.header("2. Tải lên ảnh nền mới")
uploaded_background_file = st.sidebar.file_uploader(
    "Chọn ảnh nền mới (JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    key="background_uploader",
)
raw_background_image_np = None  # Lưu ảnh nền chưa resize
if uploaded_background_file is not None:
    bytes_data_bg = uploaded_background_file.getvalue()
    background_image_pil = Image.open(io.BytesIO(bytes_data_bg)).convert("RGB")
    raw_background_image_np = np.array(background_image_pil)
    raw_background_image_np = cv2.cvtColor(raw_background_image_np, cv2.COLOR_RGB2BGR)
    st.sidebar.image(
        background_image_pil,
        caption="Ảnh nền mới (chưa xử lý)",
        use_container_width=True,
    )
    st.write(f"Kích thước ảnh nền ban đầu: {raw_background_image_np.shape}")


# --- XỬ LÝ RESIZE ẢNH NẾU KÍCH THƯỚC KHÔNG ĐỒNG NHẤT ---
processed_object_image_np = None
processed_background_image_np = None

# Chỉ thực hiện resize nếu cả hai ảnh đều đã được tải lên
if raw_object_image_np is not None and raw_background_image_np is not None:
    h1, w1 = raw_object_image_np.shape[:2]
    h2, w2 = raw_background_image_np.shape[:2]

    if h1 != h2 or w1 != w2:
        st.info(
            "Kích thước hai ảnh không đồng nhất. Đang resize ảnh lớn hơn bằng kích thước ảnh nhỏ hơn..."
        )
        if h1 * w1 > h2 * w2:  # Ảnh object lớn hơn (về tổng số pixel)
            processed_object_image_np = cv2.resize(
                raw_object_image_np, (w2, h2), interpolation=cv2.INTER_AREA
            )
            processed_background_image_np = (
                raw_background_image_np  # Ảnh nền giữ nguyên
            )
            st.write(f"Ảnh vật thể được resize từ {w1}x{h1} xuống {w2}x{h2}")
        else:  # Ảnh background lớn hơn hoặc bằng
            processed_background_image_np = cv2.resize(
                raw_background_image_np, (w1, h1), interpolation=cv2.INTER_AREA
            )
            processed_object_image_np = raw_object_image_np  # Ảnh vật thể giữ nguyên
            st.write(f"Ảnh nền được resize từ {w2}x{h2} xuống {w1}x{h1}")
    else:  # Kích thước bằng nhau
        processed_object_image_np = raw_object_image_np
        processed_background_image_np = raw_background_image_np
        st.info("Kích thước hai ảnh đã đồng nhất.")

    st.subheader("Ảnh sau khi xử lý kích thước:")
    col_proc1, col_proc2 = st.columns(2)
    if processed_object_image_np is not None:
        col_proc1.image(
            cv2.cvtColor(processed_object_image_np, cv2.COLOR_BGR2RGB),
            caption=f"Ảnh vật thể ({processed_object_image_np.shape[1]}x{processed_object_image_np.shape[0]})",
            use_container_width=True,
        )
    if processed_background_image_np is not None:
        col_proc2.image(
            cv2.cvtColor(processed_background_image_np, cv2.COLOR_BGR2RGB),
            caption=f"Ảnh nền ({processed_background_image_np.shape[1]}x{processed_background_image_np.shape[0]})",
            use_container_width=True,
        )

# --- Nhập Bounding Box ---
st.sidebar.header("3. Nhập Tọa độ Bounding Box")
st.sidebar.info(
    "Bạn cần quan sát ảnh vật thể ĐÃ XỬ LÝ KÍCH THƯỚC và nhập tọa độ (x, y, width, height) của hình chữ nhật bao quanh vật thể."
)
col1, col2 = st.sidebar.columns(2)
with col1:
    # Set max_value cho bbox_x, bbox_y để tránh nhập quá kích thước ảnh
    max_x = (
        processed_object_image_np.shape[1] - 1
        if processed_object_image_np is not None
        else 1000
    )
    max_y = (
        processed_object_image_np.shape[0] - 1
        if processed_object_image_np is not None
        else 1000
    )

    bbox_x = st.number_input(
        "X (điểm bắt đầu ngang)", min_value=0, max_value=max_x, value=0, key="bbox_x"
    )
    bbox_y = st.number_input(
        "Y (điểm bắt đầu dọc)", min_value=0, max_value=max_y, value=0, key="bbox_y"
    )
with col2:
    # Set max_value cho bbox_w, bbox_h dựa trên kích thước ảnh và bbox_x, bbox_y
    max_w = max_x + 1 - bbox_x if processed_object_image_np is not None else 1000
    max_h = max_y + 1 - bbox_y if processed_object_image_np is not None else 1000

    bbox_w = st.number_input(
        "Chiều rộng", min_value=1, max_value=max_w, value=100, key="bbox_w"
    )
    bbox_h = st.number_input(
        "Chiều cao", min_value=1, max_value=max_h, value=100, key="bbox_h"
    )

bbox = (bbox_x, bbox_y, bbox_w, bbox_h)


# --- Nút xử lý ---
st.sidebar.header("4. Thực hiện ghép ảnh")
if st.sidebar.button("Ghép ảnh", type="primary"):
    if processed_object_image_np is None or processed_background_image_np is None:
        st.error("Vui lòng tải lên cả ảnh có vật thể và ảnh nền mới trước.")
    elif bbox_w <= 0 or bbox_h <= 0:
        st.error("Kích thước Bounding Box (chiều rộng và chiều cao) phải lớn hơn 0.")
    elif bbox_x < 0 or bbox_y < 0:
        st.error("Tọa độ X và Y của Bounding Box không được âm.")
    elif (  # Kiểm tra bbox có nằm hoàn toàn trong ảnh đã xử lý không
        bbox_x + bbox_w > processed_object_image_np.shape[1]
        or bbox_y + bbox_h > processed_object_image_np.shape[0]
    ):
        st.error(
            "Bounding Box vượt quá kích thước của ảnh đã xử lý. Vui lòng điều chỉnh lại."
        )
    else:
        with st.spinner("Đang xử lý..."):
            final_image_np, object_bgr, mask2 = apply_grabcut_and_blend(
                processed_object_image_np, bbox, processed_background_image_np
            )

            if final_image_np is not None:  # Kiểm tra kết quả từ hàm GrabCut
                st.success("Xử lý hoàn tất!")

                st.header("Kết quả:")
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.image(
                        cv2.cvtColor(object_bgr, cv2.COLOR_BGR2RGB),
                        caption="Vật thể đã tách (trên nền đen)",
                        use_container_width=True,
                    )
                with col_res2:
                    st.image(
                        mask2, caption="Mask của vật thể", use_container_width=True
                    )
                with col_res3:
                    st.image(
                        cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB),
                        caption="Ảnh đã ghép",
                        use_container_width=True,
                    )

                # Tùy chọn tải xuống
                is_success, buffer = cv2.imencode(".png", final_image_np)
                if is_success:
                    st.download_button(
                        label="Tải ảnh kết quả",
                        data=buffer.tobytes(),
                        file_name="ket_qua_ghep_anh.png",
                        mime="image/png",
                    )
            else:
                st.error(
                    "Đã xảy ra lỗi trong quá trình GrabCut. Vui lòng kiểm tra lại Bounding Box."
                )
