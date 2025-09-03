-- Fixed MySQL schema (with commas).
SET NAMES utf8mb4;
SET time_zone = '+00:00';

-- Table `app_product`
DROP TABLE IF EXISTS `app_product`;
CREATE TABLE `app_product` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL,
  `image_url` VARCHAR(200) NOT NULL,
  `spec` TEXT NOT NULL,
  `material` VARCHAR(255) NOT NULL,
  `esg_report_url` VARCHAR(200) NOT NULL,
  `external_id` VARCHAR(32) NULL,
  `url` VARCHAR(200) NOT NULL,
  `composition_parts` TEXT NULL,
  `details_bullets` TEXT NULL,
  `details_intro` TEXT NOT NULL,
  `fit` VARCHAR(64) NOT NULL,
  `impact` TEXT NULL,
  `made_in` VARCHAR(128) NOT NULL,
  `price` DOUBLE NULL,
  `sustainable_detail` TEXT NOT NULL,
  `sustainable_icons` TEXT NULL,
  `category` VARCHAR(64) NOT NULL,
  `color_detail` VARCHAR(64) NOT NULL,
  `color` VARCHAR(64) NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `app_product_external_id_ce7acc55` ON `app_product` (`external_id`);

-- Table `auth_group`
DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE `auth_group` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(150) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table `auth_group_permissions`
DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE `auth_group_permissions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `group_id` BIGINT NOT NULL,
  `permission_id` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `auth_group_permissions_permission_id_84c5c92e` ON `auth_group_permissions` (`permission_id`);
CREATE INDEX `auth_group_permissions_group_id_b120cbf9` ON `auth_group_permissions` (`group_id`);
CREATE UNIQUE INDEX `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` ON `auth_group_permissions` (`group_id`, `permission_id`);

-- Table `auth_permission`
DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE `auth_permission` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `content_type_id` BIGINT NOT NULL,
  `codename` VARCHAR(100) NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `auth_permission_content_type_id_2f476e4b` ON `auth_permission` (`content_type_id`);
CREATE UNIQUE INDEX `auth_permission_content_type_id_codename_01ab375a_uniq` ON `auth_permission` (`content_type_id`, `codename`);

-- Table `auth_user`
DROP TABLE IF EXISTS `auth_user`;
CREATE TABLE `auth_user` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `password` VARCHAR(128) NOT NULL,
  `last_login` DATETIME(6) NULL,
  `is_superuser` TINYINT(1) NOT NULL,
  `username` VARCHAR(150) NOT NULL,
  `last_name` VARCHAR(150) NOT NULL,
  `email` VARCHAR(254) NOT NULL,
  `is_staff` TINYINT(1) NOT NULL,
  `is_active` TINYINT(1) NOT NULL,
  `date_joined` DATETIME(6) NOT NULL,
  `first_name` VARCHAR(150) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table `auth_user_groups`
DROP TABLE IF EXISTS `auth_user_groups`;
CREATE TABLE `auth_user_groups` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `user_id` BIGINT NOT NULL,
  `group_id` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `auth_user_groups_group_id_97559544` ON `auth_user_groups` (`group_id`);
CREATE INDEX `auth_user_groups_user_id_6a12ed8b` ON `auth_user_groups` (`user_id`);
CREATE UNIQUE INDEX `auth_user_groups_user_id_group_id_94350c0c_uniq` ON `auth_user_groups` (`user_id`, `group_id`);

-- Table `auth_user_user_permissions`
DROP TABLE IF EXISTS `auth_user_user_permissions`;
CREATE TABLE `auth_user_user_permissions` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `user_id` BIGINT NOT NULL,
  `permission_id` BIGINT NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `auth_user_user_permissions_permission_id_1fbb5f2c` ON `auth_user_user_permissions` (`permission_id`);
CREATE INDEX `auth_user_user_permissions_user_id_a95ead1b` ON `auth_user_user_permissions` (`user_id`);
CREATE UNIQUE INDEX `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` ON `auth_user_user_permissions` (`user_id`, `permission_id`);

-- Table `django_admin_log`
DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE `django_admin_log` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `object_id` TEXT NULL,
  `object_repr` VARCHAR(200) NOT NULL,
  `action_flag` INT NOT NULL,
  `change_message` TEXT NOT NULL,
  `content_type_id` BIGINT NULL,
  `user_id` BIGINT NOT NULL,
  `action_time` DATETIME(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `django_admin_log_user_id_c564eba6` ON `django_admin_log` (`user_id`);
CREATE INDEX `django_admin_log_content_type_id_c4bce8eb` ON `django_admin_log` (`content_type_id`);

-- Table `django_content_type`
DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE `django_content_type` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `app_label` VARCHAR(100) NOT NULL,
  `model` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE UNIQUE INDEX `django_content_type_app_label_model_76bd3d3b_uniq` ON `django_content_type` (`app_label`, `model`);

-- Table `django_migrations`
DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE `django_migrations` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `app` VARCHAR(255) NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `applied` DATETIME(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table `django_session`
DROP TABLE IF EXISTS `django_session`;
CREATE TABLE `django_session` (
  `session_key` VARCHAR(40) NOT NULL,
  `session_data` TEXT NOT NULL,
  `expire_date` DATETIME(6) NOT NULL,
  PRIMARY KEY (`session_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX `django_session_expire_date_a5c62663` ON `django_session` (`expire_date`);