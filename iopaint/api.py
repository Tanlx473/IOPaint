"""基于 FastAPI 的后端，提供 IOPaint 所需的接口。

该模块向前端暴露 REST 与 Socket.IO 接口，处理图片上传、模型切换
以及插件执行等功能。

FastAPI backend exposing REST and Socket.IO endpoints for image
upload, model selection and plugin execution.

"""

import asyncio
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np
import socketio
import torch

try:
    # 禁用 PyTorch 一些可能带来额外开销的 JIT 与 Fuser 功能
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_profiling_mode(False)
except:  # noqa: E722 - 保持与原库兼容
    # 某些环境可能不存在上述接口，忽略异常即可
    pass

import uvicorn
from PIL import Image
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from socketio import AsyncServer

from iopaint.file_manager import FileManager
from iopaint.helper import (
    load_img,
    decode_base64_to_image,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
    gen_frontend_mask,
    adjust_mask,
)
from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.plugins import build_plugins, RealESRGANUpscaler, InteractiveSeg
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.plugins.remove_bg import RemoveBG
from iopaint.schema import (
    GenInfoResponse,
    ApiConfig,
    ServerConfigResponse,
    SwitchModelRequest,
    InpaintRequest,
    RunPluginRequest,
    SDSampler,
    PluginInfo,
    AdjustMaskRequest,
    RemoveBGModel,
    SwitchPluginModelRequest,
    ModelInfo,
    InteractiveSegModel,
    RealESRGANModel,
)

CURRENT_DIR = Path(__file__).parent.absolute().resolve()
WEB_APP_DIR = CURRENT_DIR / "web_app"


def api_middleware(app: FastAPI):
    """为 FastAPI 应用添加通用中间件和异常处理。"""

    # 控制是否启用 rich 格式的异常输出
    rich_available = False
    try:
        # 若设置 WEBUI_RICH_EXCEPTIONS 环境变量，则启用 rich 格式化输出
        if os.environ.get("WEBUI_RICH_EXCEPTIONS", None) is not None:
            import anyio  # 放入 suppress 列表，避免打印不必要的栈信息
            import starlette  # 同上，仅用于异常抑制
            from rich.console import Console

            # 创建 rich 控制台用于美化异常输出
            console = Console()
            rich_available = True
    except Exception:
        # rich 未安装或导入失败时，直接忽略，继续使用默认日志
        pass

    def handle_exception(request: Request, e: Exception):
        """统一处理接口异常，生成格式化的 JSON 响应"""

        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        # HTTPException 为已知错误，避免重复打印回溯
        if not isinstance(
            e, HTTPException
        ):
            # 打印错误基本信息并根据是否支持 rich 决定输出方式
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                traceback.print_exc()
        return JSONResponse(
            # 使用异常自带的状态码，默认 500
            status_code=vars(e).get("status_code", 500),
            content=jsonable_encoder(err),
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        """拦截请求，在出现异常时返回统一格式。"""
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        """全局异常处理器。"""
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        """处理 FastAPI 内置的 HTTP 异常。"""
        return handle_exception(request, e)

    # 允许跨域访问，方便前端独立部署
    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_origins": ["*"],
        "allow_credentials": True,
        "expose_headers": ["X-Seed"],
    }
    app.add_middleware(CORSMiddleware, **cors_options)


# 用于在回调中发送 SocketIO 事件的全局变量
global_sio: AsyncServer = None


def diffuser_callback(pipe, step: int, timestep: int, callback_kwargs: Dict = {}):
    """在推理过程中回调，向前端实时汇报进度。"""

    # 异步发送当前步骤给前端界面
    asyncio.run(global_sio.emit("diffusion_progress", {"step": step}))
    return {}


class Api:
    """封装后端所有 API 的核心类。

    Core class that exposes all backend APIs.
    """

    def __init__(self, app: FastAPI, config: ApiConfig):
        """初始化 API，对外注册各类路由并加载模型与插件。"""

        self.app = app
        self.config = config
        # 自定义路由注册器及并发锁
        self.router = APIRouter()
        self.queue_lock = threading.Lock()
        # 应用基础中间件和异常处理
        api_middleware(self.app)

        # 初始化文件管理、插件和模型管理器
        self.file_manager = self._build_file_manager()
        self.plugins = self._build_plugins()
        self.model_manager = self._build_model_manager()

        # fmt: off
        # 注册所有 REST 接口
        self.add_api_route("/api/v1/gen-info", self.api_geninfo, methods=["POST"], response_model=GenInfoResponse)
        self.add_api_route("/api/v1/server-config", self.api_server_config, methods=["GET"],
                           response_model=ServerConfigResponse)
        self.add_api_route("/api/v1/model", self.api_current_model, methods=["GET"], response_model=ModelInfo)
        self.add_api_route("/api/v1/model", self.api_switch_model, methods=["POST"], response_model=ModelInfo)
        self.add_api_route("/api/v1/inputimage", self.api_input_image, methods=["GET"])
        self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])
        self.add_api_route("/api/v1/switch_plugin_model", self.api_switch_plugin_model, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_mask", self.api_run_plugin_gen_mask, methods=["POST"])
        self.add_api_route("/api/v1/run_plugin_gen_image", self.api_run_plugin_gen_image, methods=["POST"])
        self.add_api_route("/api/v1/samplers", self.api_samplers, methods=["GET"])
        self.add_api_route("/api/v1/adjust_mask", self.api_adjust_mask, methods=["POST"])
        self.add_api_route("/api/v1/save_image", self.api_save_image, methods=["POST"])
        self.app.mount("/", StaticFiles(directory=WEB_APP_DIR, html=True), name="assets")
        # fmt: on

        global global_sio
        # 初始化 Socket.IO 服务，用于实时通信
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.combined_asgi_app = socketio.ASGIApp(self.sio, self.app)
        self.app.mount("/ws", self.combined_asgi_app)
        global_sio = self.sio

    def add_api_route(self, path: str, endpoint, **kwargs):
        """包装 FastAPI 的路由注册方法，便于日后统一管理。"""

        return self.app.add_api_route(path, endpoint, **kwargs)

    def api_save_image(self, file: UploadFile):
        """接收上传的图片并保存到输出目录。"""

        # 使用纯文件名避免目录穿越
        safe_filename = Path(file.filename).name

        # 构造输出路径
        output_path = self.config.output_dir / safe_filename

        # 检查输出目录是否可用
        if not self.config.output_dir or not self.config.output_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Output directory not configured or doesn't exist",
            )

        # 将上传内容写入磁盘
        origin_image_bytes = file.file.read()
        with open(output_path, "wb") as fw:
            fw.write(origin_image_bytes)

    def api_current_model(self) -> ModelInfo:
        """获取当前使用的模型信息。"""

        return self.model_manager.current_model

    def api_switch_model(self, req: SwitchModelRequest) -> ModelInfo:
        """根据请求切换主模型。"""

        if req.name == self.model_manager.name:
            return self.model_manager.current_model
        self.model_manager.switch(req.name)
        return self.model_manager.current_model

    def api_switch_plugin_model(self, req: SwitchPluginModelRequest):
        """切换指定插件所使用的模型。"""

        if req.plugin_name in self.plugins:
            self.plugins[req.plugin_name].switch_model(req.model_name)
            if req.plugin_name == RemoveBG.name:
                self.config.remove_bg_model = req.model_name
            if req.plugin_name == RealESRGANUpscaler.name:
                self.config.realesrgan_model = req.model_name
            if req.plugin_name == InteractiveSeg.name:
                self.config.interactive_seg_model = req.model_name
            torch_gc()

    def api_server_config(self) -> ServerConfigResponse:
        """返回前端所需的服务器配置与插件信息。"""

        plugins = []
        for it in self.plugins.values():
            plugins.append(
                PluginInfo(
                    name=it.name,
                    support_gen_image=it.support_gen_image,
                    support_gen_mask=it.support_gen_mask,
                )
            )

        # 整合当前服务与插件的配置信息返回给前端
        return ServerConfigResponse(
            plugins=plugins,
            modelInfos=self.model_manager.scan_models(),
            removeBGModel=self.config.remove_bg_model,
            removeBGModels=RemoveBGModel.values(),
            realesrganModel=self.config.realesrgan_model,
            realesrganModels=RealESRGANModel.values(),
            interactiveSegModel=self.config.interactive_seg_model,
            interactiveSegModels=InteractiveSegModel.values(),
            enableFileManager=self.file_manager is not None,
            enableAutoSaving=self.config.output_dir is not None,
            enableControlnet=self.model_manager.enable_controlnet,
            controlnetMethod=self.model_manager.controlnet_method,
            disableModelSwitch=False,
            isDesktop=False,
            samplers=self.api_samplers(),
        )

    def api_input_image(self) -> FileResponse:
        """返回预设的输入图片。"""

        if self.config.input is None:
            raise HTTPException(status_code=200, detail="No input image configured")

        if self.config.input.is_file():
            return FileResponse(self.config.input)
        raise HTTPException(status_code=404, detail="Input image not found")

    def api_geninfo(self, file: UploadFile) -> GenInfoResponse:
        """
            解析上传图片中保存的提示词信息。
            逐行说明：api_geninfo 是 Api 类中的一个接口方法，对应路由 /api/v1/gen-info。
                    参数 file: UploadFile 表示前端上传的图片文件。返回值类型为 GenInfoResponse（定义于 schema.py，含 prompt 和 negative_prompt 两个字段）。
                    load_img(file.file.read(), return_info=True) 调用 load_img 读取上传的图片二进制数据。
                    return_info=True 使函数同时返回图片的 info 字典（即 PIL.Image 读取到的元数据）。
                    info.get("parameters", "") 在一些由 Stable Diffusion 等模型生成的 PNG 图像中，生成时的提示词会存放在元数据 parameters 字段中。若该字段不存在，默认使用空字符串。
                    .split("Negative prompt: ") Stable Diffusion 将提示词和反向提示词（Negative Prompt）放在同一字段，通常格式为：prompt文字 Negative prompt: xxx\nSteps: ...。这里按 "Negative prompt: " 分割，得到包含前半部分（正向提示词）和后半部分（反向提示词及其后续参数信息）的列表。
                    prompt = parts[0].strip()第一段即为正向提示词，去除前后空白。
                    negative_prompt = "" 初始化反向提示词为空字符串。
                    if len(parts) > 1:若分割后得到两段以上，说明元数据中确实包含 Negative prompt。
                    negative_prompt = parts[1].split("\n")[0].strip() 取第二段的第一行文本作为反向提示词（因为之后往往还有 Steps、Seed 等其他参数）。最终也去除首尾空白。
                    return GenInfoResponse(prompt=prompt, negative_prompt=negative_prompt) 构造并返回 GenInfoResponse，使前端能直接获取到相应的提示词内容。
            该方法的作用是解析上传图片（通常是经过模型生成或编辑过的 PNG）所携带的元数据，从中提取出正向/反向提示词，供前端在重新编辑或复原图片参数时使用。
        """

        _, _, info = load_img(file.file.read(), return_info=True)  
        parts = info.get("parameters", "").split("Negative prompt: ")
        prompt = parts[0].strip()
        negative_prompt = ""
        if len(parts) > 1:
            negative_prompt = parts[1].split("\n")[0].strip()
        return GenInfoResponse(prompt=prompt, negative_prompt=negative_prompt)

    def api_inpaint(self, req: InpaintRequest):
        """根据掩码对图像进行修复并返回结果。"""

        image, alpha_channel, infos, ext = decode_base64_to_image(req.image)
        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        logger.info(f"image ext: {ext}")

        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # 校验图片与掩码尺寸是否一致
        if image.shape[:2] != mask.shape[:2]:
            raise HTTPException(
                400,
                detail=f"Image size({image.shape[:2]}) and mask size({mask.shape[:2]}) not match.",
            )

        start = time.time()
        rgb_np_img = self.model_manager(image, mask, req)
        # 打印推理耗时
        logger.info(f"process time: {(time.time() - start) * 1000:.2f}ms")
        torch_gc()

        # 将模型输出转换为带透明通道的 RGBA
        rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

        res_img_bytes = pil_to_bytes(
            Image.fromarray(rgb_res),
            ext=ext,
            quality=self.config.quality,
            infos=infos,
        )

        # 通知前端任务结束
        asyncio.run(self.sio.emit("diffusion_finish"))

        return Response(
            content=res_img_bytes,
            media_type=f"image/{ext}",
            headers={"X-Seed": str(req.sd_seed)},
        )

    def api_run_plugin_gen_image(self, req: RunPluginRequest):
        """调用插件生成图像，返回处理后的结果。"""

        ext = "png"
        if req.name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        if not self.plugins[req.name].support_gen_image:
            raise HTTPException(
                status_code=422, detail="Plugin does not support output image"
            )
        rgb_np_img, alpha_channel, infos, _ = decode_base64_to_image(req.image)
        bgr_or_rgba_np_img = self.plugins[req.name].gen_image(rgb_np_img, req)
        torch_gc()

        if bgr_or_rgba_np_img.shape[2] == 4:
            rgba_np_img = bgr_or_rgba_np_img
        else:
            rgba_np_img = cv2.cvtColor(bgr_or_rgba_np_img, cv2.COLOR_BGR2RGB)
            rgba_np_img = concat_alpha_channel(rgba_np_img, alpha_channel)

        return Response(
            content=pil_to_bytes(
                Image.fromarray(rgba_np_img),
                ext=ext,
                quality=self.config.quality,
                infos=infos,
            ),
            media_type=f"image/{ext}",
        )

    def api_run_plugin_gen_mask(self, req: RunPluginRequest):
        """调用插件生成掩码。"""

        if req.name not in self.plugins:
            raise HTTPException(status_code=422, detail="Plugin not found")
        if not self.plugins[req.name].support_gen_mask:
            raise HTTPException(
                status_code=422, detail="Plugin does not support output image"
            )
        rgb_np_img, _, _, _ = decode_base64_to_image(req.image)
        bgr_or_gray_mask = self.plugins[req.name].gen_mask(rgb_np_img, req)
        torch_gc()
        res_mask = gen_frontend_mask(bgr_or_gray_mask)
        return Response(
            content=numpy_to_bytes(res_mask, "png"),
            media_type="image/png",
        )

    def api_samplers(self) -> List[str]:
        """列出可用的采样器名称。"""

        return [member.value for member in SDSampler.__members__.values()]

    def api_adjust_mask(self, req: AdjustMaskRequest):
        """在前端请求下对掩码做膨胀/腐蚀等形态学处理。"""

        mask, _, _, _ = decode_base64_to_image(req.mask, gray=True)
        mask = adjust_mask(mask, req.kernel_size, req.operate)
        return Response(content=numpy_to_bytes(mask, "png"), media_type="image/png")

    def launch(self):
        """启动 Uvicorn 服务器。"""

        self.app.include_router(self.router)
        uvicorn.run(
            self.combined_asgi_app,
            host=self.config.host,
            port=self.config.port,
            timeout_keep_alive=999999999,
        )

    def _build_file_manager(self) -> Optional[FileManager]:
        """根据配置生成文件管理器，用于读取输入与保存输出。"""

        if self.config.input and self.config.input.is_dir():
            logger.info(
                f"Input is directory, initialize file manager {self.config.input}"
            )

            return FileManager(
                app=self.app,
                input_dir=self.config.input,
                mask_dir=self.config.mask_dir,
                output_dir=self.config.output_dir,
            )
        return None

    def _build_plugins(self) -> Dict[str, BasePlugin]:
        """构建所有已启用的插件实例。"""

        return build_plugins(
            self.config.enable_interactive_seg,
            self.config.interactive_seg_model,
            self.config.interactive_seg_device,
            self.config.enable_remove_bg,
            self.config.remove_bg_device,
            self.config.remove_bg_model,
            self.config.enable_anime_seg,
            self.config.enable_realesrgan,
            self.config.realesrgan_device,
            self.config.realesrgan_model,
            self.config.enable_gfpgan,
            self.config.gfpgan_device,
            self.config.enable_restoreformer,
            self.config.restoreformer_device,
            self.config.no_half,
        )

    def _build_model_manager(self):
        """初始化模型管理器。"""

        return ModelManager(
            name=self.config.model,
            device=torch.device(self.config.device),
            no_half=self.config.no_half,
            low_mem=self.config.low_mem,
            disable_nsfw=self.config.disable_nsfw_checker,
            sd_cpu_textencoder=self.config.cpu_textencoder,
            local_files_only=self.config.local_files_only,
            cpu_offload=self.config.cpu_offload,
            callback=diffuser_callback,
        )
