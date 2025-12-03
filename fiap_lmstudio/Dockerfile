FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

COPY ../AppImage* /data/allm/
COPY ./keyboard /etc/default/keyboard
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && add-apt-repository ppa:apt-fast/stable && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y apt-fast


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-fast install -y kmod keyboard-configuration console-setup tzdata dbus x11-utils x11-xserver-utils libgl1-mesa-glx
RUN dpkg-reconfigure -f noninteractive keyboard-configuration
RUN apt-get update &&  apt-fast install -y libfuse2 kmod fuse libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgtk-3-0 libgbm1 libasound2 xserver-xorg xvfb x11vnc python3 python3-pip git xdotool wmctrl net-tools
RUN pip3 install websockify
RUN git clone https://github.com/novnc/noVNC.git /opt/noVNC
RUN mkdir -p /root/.vnc && x11vnc -storepasswd lmstudio123 /root/.vnc/passwd
ENV DISPLAY=:99
RUN chmod ugo+x /data/allm/*.AppImage
RUN /data/allm/*.AppImage --appimage-extract
RUN cd /squashfs-root/
RUN chown root:root /squashfs-root/chrome-sandbox
RUN chmod 4755 /squashfs-root/chrome-sandbox
# Criar link simbólico para CLI do LM Studio
RUN ln -s /squashfs-root/resources/app/.webpack/lms /usr/local/bin/lms
COPY start_services.sh /start_services.sh
COPY http-server-config.json /http-server-config.json
COPY index.html /opt/noVNC/index.html
RUN chmod +x /start_services.sh

CMD ["/start_services.sh"]