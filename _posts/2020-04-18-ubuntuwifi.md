---
layout: single
title: "우분투OS 무선네트워크 드라이버 문제해결"
date: 2020-04-18
comments: true
categories: 
    [LINUX]
tags:
    [ubuntu, Asus Vivobook, IWLWIFI]
toc: false
publish: true
classes: wide
comments: true
permalink: categories/csdev/linux
---

지난달 여자친구가 사용할 새로운 노트북으로 ASUS Vivobook X412 을 수령했습니다. 해당 모델같은 경우 20년 2월에 갓등록된 새로운 14인치 화면에 10세대 코어i5를 담고있는 가격대비 상당한 가성비를 갖고있는 제품이라 기대가 많이 되더라구요. 쿼드코어 1.6GHz 에 터보부스트를 사용하면 4.2GHz 까지 올라는 성능이라 기본적인 데이터 프로세싱이 가능한 가벼운 노트북으로서 적합해보이는 모델이라 추천했습니다. 구매부터 우분투 18.04 설치까지는 원활히 진행되었는데 문제는 무선네트워크 설정을 위한 어댑터 드라이버를 찾는데 발생했습니다.

출시되지 얼마되지않은 노트북과 같은 경우 Linux 5.30–40-generic x86_64버전으로서 가장 최신 커널로 업데이트 했음에도 불구하고 어댑터 드라이버를 자동으로 인식하지못하더군요. 구글에 찾아봐도 잘 나오지않았지만 레딧에서 이미 비슷한 증상을 겪고있는 포스트를 찾을 수 있었습니다.

최근에 출시한 노트북에 우분투를 설치하는 경우 이더넷포트가 없어 랜선을 통한 연결도 안되고 커널상에도 최신 드라이버가 없어서 와이파이와 블루투스도 안되는 진퇴양난/캐치-22 상환에 마주하게됩니다. 다행히도 스마트폰과 연결해 USB 연결을 통한 데이터 테더링을 통해 임시적으로 인터넷 연결을 할 수 있었습니다.

비슷한 문제에 직면한 분들을 위해 솔루션을 남겨놓습니다:

1. ```ifconfig```을 통해 무선랜 연결상태를 체크하고 인식되는 기기가 없는지를 우선적으로 체크합니다

2. ```uname -srm``` 을 통해 리눅스 커널 버전을 확인합니다. 가장 최신 버전의 리눅스 커널같은 경우에는 보통 유니버설 무선랜 드라이버가 기본적으로 포함되어있더군요. 필요하다면 ```sudo apt-get dist-upgrade``` 를 통해 커널을 업데이트 합니다. 커널 버전 5.0 이후부터는 내장어댑터에 보편적으로 사용되고있는 dkms RTL8168 드라이버 (Realtek 802.11n WLAN)가 추가적으로 필요하지않습니다.

3. 우분투OS 18.04를 19.04(베타)로 업그레이드해서 새로운 커널 체인을 적용합니다. 검증되지않은 버전이라 주의가 필요하지만 일부 인텔 WIFI 리눅스 드라이버 ‘IWLWIFI’가 커널 5.1부터 적용될 예정이기 때문에 OS 업데이트가 필요합니다.

4. 업데이트 후에도 네트워크 어댑터 인식이 불가능하다면 IWLWIFI (Intel WiFi Linux Driver) 의 새로운 버전을 다음 커맨드로 설치해주시면 됩니다:


```
sudo apt update
sudo apt install git build-essential
git clone https://git.kernel.org/pub/scm/linux/kernel/git/iwlwifi/backport-iwlwifi.git
cd backport-iwlwifi/
make defconfig-iwlwifi-public
sed -i 's/CPTCFG_IWLMVM_VENDOR_CMDS=y/# CPTCFG_IWLMVM_VENDOR_CMDS is not set/' .config
make -j4
sudo make install
sudo modprobe iwlwifi
```

Reference: [link] https://askubuntu.com/questions/1156167/unable-to-get-wifi-adapter-working-clean-19-04-install-network-unclaimed